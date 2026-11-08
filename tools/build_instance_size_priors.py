"""Build size modes from the configured *unrepeated* training annotation set."""
import argparse
import copy
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings

from mmdet3d.registry import DATASETS
from igdet.size_priors import build_size_priors


def build_from_config(config, output, max_modes=4, min_samples_per_mode=8, seed=0):
    cfg = Config.fromfile(config)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    dataset_cfg = copy.deepcopy(cfg.train_dataloader.dataset)
    while dataset_cfg.get('type') == 'RepeatDataset':
        dataset_cfg = dataset_cfg['dataset']
    if dataset_cfg.get('type') != 'MultiViewScanNetDataset':
        raise ValueError('This builder currently supports MultiViewScanNetDataset only')
    dataset_cfg.update(pipeline=[], ann_file_2d=None, test_mode=False,
                       filter_empty_gt=False)
    dataset = DATASETS.build(dataset_cfg)
    classes = list(cfg.model.g_dino_cfg.semantic_classes)
    dataset_classes = list(dataset.metainfo['classes'])
    def canonical(name):
        return name.replace(' ', '').replace('showercurtrain', 'showercurtain')
    if [canonical(c) for c in classes] != [canonical(c) for c in dataset_classes]:
        raise ValueError('Dataset and semantic class order differ')
    all_sizes, all_labels, seen = [], [], set()
    for index in range(len(dataset)):
        info = dataset.get_data_info(index)
        scene = info['scene_id']
        if scene in seen:
            raise ValueError(f'Duplicate training scene {scene!r}; use an unrepeated annotation set')
        seen.add(scene)
        annotation = dataset.get_ann_info(index)
        sizes = torch.as_tensor(annotation['gt_bboxes_3d'].tensor[:, 3:6])
        labels = torch.as_tensor(annotation['gt_labels_3d'], dtype=torch.long)
        keep = labels >= 0
        all_sizes.append(sizes[keep])
        all_labels.append(labels[keep])
    if not all_sizes:
        raise ValueError('Training dataset is empty')
    artifact = build_size_priors(torch.cat(all_sizes), torch.cat(all_labels), classes,
                                 max_modes, min_samples_per_mode, seed)
    annotation_path = Path(dataset.ann_file)
    artifact.update(source_annotation=str(annotation_path.resolve()),
                    source_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
                    source_config=str(Path(config).resolve()),
                    dataset_classes=dataset_classes, scene_count=len(seen))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-modes', type=int, default=4)
    parser.add_argument('--min-samples-per-mode', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    artifact = build_from_config(args.config, args.output, args.max_modes,
                                  args.min_samples_per_mode, args.seed)
    print(f"Saved {int(artifact['valid_modes'].sum())} modes from "
          f"{artifact['scene_count']} training scenes to {args.output}")


if __name__ == '__main__':
    main()
