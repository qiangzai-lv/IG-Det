"""Standalone check of the exact rotated IoU/GIoU used by IG-Det.

Loads rotated_geometry.py directly: no MMDetection, VGGT, dataset or checkpoint
is required. NPU runs the native PyTorch implementation, not a CUDA extension.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import torch


def load_geometry():
    path = Path(__file__).resolve().parents[2] / 'igdet/rotated_geometry.py'
    spec = importlib.util.spec_from_file_location('igdet_rotated_geometry_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_device(name):
    if name.startswith('npu'):
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            raise RuntimeError('torch_npu is installed but no NPU is available')
        torch.npu.set_device(name)
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        torch.cuda.set_device(device)
    return device


def synchronize(device):
    if device.type == 'npu':
        torch.npu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def sample_boxes(count, seed):
    rng = torch.Generator().manual_seed(seed)
    boxes = torch.randn(count, 7, generator=rng)
    boxes[:, :3] *= .7
    boxes[:, 3:6] = boxes[:, 3:6].abs() + .4
    return boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='npu:0', help='npu:0, cuda:0 or cpu; never silently falls back')
    parser.add_argument('--boxes', type=int, default=64)
    parser.add_argument('--targets', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('work_dirs/rotated_iou_check.json'))
    args = parser.parse_args()
    if min(args.boxes, args.targets, args.iterations) < 1:
        parser.error('boxes, targets and iterations must be positive')
    report = dict(requested_device=args.device, torch=torch.__version__, python=platform.python_version(), checks=[])
    current = 'device initialization'
    try:
        geometry = load_geometry()
        device = choose_device(args.device)
        report['device'] = str(device)
        if device.type == 'npu':
            import torch_npu
            report['torch_npu'] = torch_npu.__version__
            report['device_name'] = torch.npu.get_device_name(device.index or 0)
        elif device.type == 'cuda':
            report['device_name'] = torch.cuda.get_device_name(device)
        else:
            report['device_name'] = platform.processor()
        print(json.dumps(report, ensure_ascii=False), flush=True)

        def check(name, actual, expected, atol=2e-4, rtol=2e-4):
            if actual.device.type != device.type:
                raise AssertionError(f'{name}: output is on {actual.device}, expected {device}')
            actual = actual.detach().float().cpu()
            expected = expected.detach().float().cpu()
            if not torch.isfinite(actual).all():
                raise AssertionError(f'{name}: non-finite output')
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            error = float((actual-expected).abs().max()) if actual.numel() else 0.
            report['checks'].append(dict(name=name, passed=True, max_abs_error=error))
            print(f'PASS {name}: max_abs_error={error:.6g}', flush=True)

        cpu_a, cpu_b = sample_boxes(args.boxes, 12), sample_boxes(args.targets, 31)
        a, b = cpu_a.to(device), cpu_b.to(device)
        for mode in ['iou', 'giou']:
            current = f'pairwise_{mode}'
            print(f'RUN {current}', flush=True)
            with torch.no_grad():
                expected = geometry.rotated_overlaps(cpu_a, cpu_b, mode=mode)
                result = geometry.rotated_overlaps(a, b, mode=mode)
            check(current, result, expected)
            current = f'aligned_{mode}'
            count = min(len(a), len(b))
            check(current, geometry.rotated_overlaps(a[:count], b[:count], mode=mode, aligned=True),
                  geometry.rotated_overlaps(cpu_a[:count], cpu_b[:count], mode=mode, aligned=True))

        current = 'giou_backward'
        print(f'RUN {current}', flush=True)
        count = min(len(a), len(b))
        cpu_train = cpu_a[:count].clone().requires_grad_()
        train = a[:count].clone().requires_grad_()
        cpu_loss = (1-geometry.rotated_overlaps(cpu_train,cpu_b[:count],mode='giou',aligned=True)).mean()
        loss = (1-geometry.rotated_overlaps(train,b[:count],mode='giou',aligned=True)).mean()
        cpu_loss.backward()
        loss.backward()
        synchronize(device)
        check('giou_loss',loss,cpu_loss)
        check('giou_backward',train.grad,cpu_train.grad,atol=2e-3,rtol=2e-3)
        if not train.grad.abs().sum() > 0 or not train.grad[:,6].abs().sum() > 0:
            raise AssertionError('Missing geometry/yaw gradient')

        current = 'boundary_and_empty_cases'
        fixed = torch.tensor([[0.,0,0,4,1,2,.3],[0.,0,0,2,2,2,0],
                              [0.,0,0,2,2,2,0],[0.,0,0,.01,.01,.01,.7]])
        other = fixed.clone()
        other[0,6] += torch.pi  # equivalent orientation
        other[1,0] = 2.        # touching, zero intersection
        other[2,2] = 10.       # overlapping BEV, disjoint volume
        for mode in ['iou','giou']:
            check(f'boundary_{mode}',geometry.rotated_overlaps(fixed.to(device),other.to(device),mode=mode,aligned=True),
                  geometry.rotated_overlaps(fixed,other,mode=mode,aligned=True))
        check('empty_pairwise',geometry.rotated_overlaps(a[:0],b),torch.empty(0,len(b)))
        for dtype in [torch.float16,torch.bfloat16]:
            current = f'{dtype}_input'
            # Expected uses identically quantized inputs. Geometry itself is FP32.
            aq,bq=cpu_a.to(dtype),cpu_b.to(dtype)
            result=geometry.rotated_overlaps(aq.to(device),bq.to(device),mode='giou')
            if result.dtype != torch.float32:
                raise AssertionError('Geometry must promote half inputs to FP32')
            check(current,result,geometry.rotated_overlaps(aq,bq,mode='giou'))

        current = 'volume_and_bev_nms'
        fixed = torch.tensor([[0.,0,0,2,1,1,.4],[0.,0,4,2,1,1,.4],[0.,0,0,2,1,1,.4]])
        scores=torch.tensor([.9,.8,.7]);labels=torch.tensor([0,0,1])
        for bev in [False,True]:
            result=geometry.rotated_nms(fixed.to(device),scores.to(device),labels.to(device),.5,bev=bev)
            check(f'nms_bev={bev}',result,geometry.rotated_nms(fixed,scores,labels,.5,bev=bev),atol=0,rtol=0)

        current = 'synchronized_timing'
        timing={}
        for mode in ['iou','giou']:
            synchronize(device)
            started=time.perf_counter()
            for _ in range(args.iterations):
                with torch.no_grad():geometry.rotated_overlaps(a,b,mode=mode)
                synchronize(device)
            timing[f'pairwise_{mode}_ms']=(time.perf_counter()-started)*1000/args.iterations
        report.update(passed=True,timing=timing,shape=[args.boxes,args.targets])
        print('TIMING',json.dumps(timing),flush=True)
        print('ALL CHECKS PASSED',flush=True)
    except Exception as error:
        report.update(passed=False,failed_at=current,error=repr(error),traceback=traceback.format_exc())
        print(f'FAILED at {current}: {error}',flush=True)
        traceback.print_exc()
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
        print(f'Report: {args.output.resolve()}',flush=True)
    return 0 if report.get('passed') else 1


if __name__=='__main__':
    sys.exit(main())
