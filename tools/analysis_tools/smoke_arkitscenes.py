"""Small real-data LoRA forward/backward check, no full training schedule.

2D writes a compact complete 2D checkpoint (detector + LoRA + pyramid),
which 3D can consume through the normal retention loader.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.optim import OptimWrapper
from mmdet3d.registry import DATASETS,MODELS
from mmdet3d.utils import register_all_modules
import igdet


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['2d','3d'],required=True)
    parser.add_argument('--views',type=int,default=2)
    parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--checkpoint',default='work_dirs/arkitscenes_2d_smoke/iter_2.pth')
    args=parser.parse_args()
    register_all_modules()
    torch.manual_seed(0)
    cfg=Config.fromfile(f'configs/igdet/igdet_arkitscenes{"_2d" if args.stage=="2d" else ""}.py')
    if args.stage=='3d':
        cfg.model.g_dino_cfg.grounding_dino_checkpoint=args.checkpoint
        cfg.model.two_d_retention_cfg.checkpoint=args.checkpoint
    dataset_cfg=cfg.train_dataloader.dataset.dataset
    view_pipeline=next(item for item in dataset_cfg.pipeline if item.type=='MultiViewPipeline')
    view_pipeline.n_images=args.views
    view_pipeline.loading='uniform'
    out=Path(f'work_dirs/arkitscenes_{args.stage}_smoke');out.mkdir(parents=True,exist_ok=True)
    model=MODELS.build(cfg.model)
    model.init_weights()
    model.cuda().train()
    dataset=DATASETS.build(dataset_cfg)
    optimizer=OptimWrapper(torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5),
                          clip_grad=dict(max_norm=1.,norm_type=2))
    adapters={n:p for n,p in model.named_parameters() if '.lora_b.' in n}
    assert adapters and all(p.requires_grad for p in adapters.values())
    tracked=next(iter(adapters));before=adapters[tracked].detach().clone()
    frozen={n:p.detach().clone() for n,p in model.named_parameters()
            if not p.requires_grad and (n.endswith('aggregator.patch_embed.proj.weight') or n.endswith('bbox_head.instance_projection.0.weight'))}
    frozen_2d=[p for p in model.semantic_encoder.model.parameters() if not p.requires_grad]
    seen_gradients={}
    def record_gradient(name):
        def record(gradient):
            seen_gradients[name]=(bool(torch.isfinite(gradient).all()),float(gradient.abs().sum()))
        return record
    handles=[p.register_hook(record_gradient(n)) for n,p in adapters.items()]
    reports=[]
    for step in range(args.steps):
        data=pseudo_collate([dataset[step%len(dataset)]])
        seen_gradients.clear()
        logs=model.train_step(data,optimizer)
        report={k:float(v.detach()) if isinstance(v,torch.Tensor) else float(v) for k,v in logs.items()}
        assert all(torch.isfinite(torch.tensor(v)) for v in report.values())
        assert seen_gradients and all(value[0] for value in seen_gradients.values())
        assert any(value[1]>0 for value in seen_gradients.values())
        assert all(p.grad is None for p in frozen_2d)
        reports.append(report);print('STEP',step,json.dumps(report),flush=True)
    assert not torch.equal(before,adapters[tracked].detach())
    current=dict(model.named_parameters())
    assert all(torch.equal(p,current[n].detach()) for n,p in frozen.items())
    if args.stage=='2d':
        state={n:v.cpu() for n,v in model.state_dict().items() if n.startswith(('semantic_encoder.model.','detection_feature_projector.')) or '.lora_a.' in n or '.lora_b.' in n}
        torch.save(dict(state_dict=state,meta=dict(cfg=cfg.pretty_text)),out/f'iter_{args.steps}.pth')
    else:
        model.eval()
        val_cfg=cfg.val_dataloader.dataset
        next(item for item in val_cfg.pipeline if item.type=='MultiViewPipeline').n_images=args.views
        val_dataset=DATASETS.build(val_cfg)
        with torch.no_grad():
            data=model.data_preprocessor(pseudo_collate([val_dataset[0]]),False)
            predictions=model(**data,mode='predict')
        assert predictions[0].pred_instances_3d.bboxes_3d.tensor.shape[1]==7
        reports.append(dict(predictions=len(predictions[0].pred_instances_3d.bboxes_3d)))
    (out/'result.json').write_text(json.dumps(dict(stage=args.stage,views=args.views,steps=reports,lora_updated=True,frozen_unchanged=True),indent=2))
    print('SMOKE PASSED',args.stage,flush=True)


if __name__=='__main__':main()
