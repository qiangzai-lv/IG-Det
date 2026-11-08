"""Deterministic log-space size clustering, using training boxes only."""
import torch


def build_size_priors(sizes, labels, classes, max_modes=4,
                      min_samples_per_mode=8, seed=0):
    sizes = torch.as_tensor(sizes, dtype=torch.float32, device='cpu')
    labels = torch.as_tensor(labels, dtype=torch.long, device='cpu')
    if sizes.ndim != 2 or sizes.shape[-1] != 3 or labels.shape != sizes.shape[:1]:
        raise ValueError('expected sizes [N, 3] and labels [N]')
    if max_modes < 1 or min_samples_per_mode < 1:
        raise ValueError('mode counts and minimum sample counts must be positive')
    if not torch.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError('training size priors require finite positive box sizes')
    if ((labels < 0) | (labels >= len(classes))).any():
        raise ValueError('size-prior label is outside the class mapping')
    priors = torch.ones(len(classes), max_modes, 3)
    valid = torch.zeros(len(classes), max_modes, dtype=torch.bool)
    counts = torch.zeros(len(classes), max_modes, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    for category in range(len(classes)):
        values = sizes[labels == category].log()
        if len(values) == 0:
            raise ValueError(f'No training boxes for category {classes[category]!r}')
        unique = values.unique(dim=0)
        modes = min(max_modes, len(unique), max(1, len(values) // min_samples_per_mode))
        centers = [unique[torch.randint(len(unique), (), generator=generator)]]
        for _ in range(1, modes):
            distances = torch.cdist(unique, torch.stack(centers)).square().amin(-1)
            centers.append(unique[distances.argmax()])
        centers = torch.stack(centers)
        for _ in range(100):
            assignment = torch.cdist(values, centers).argmin(-1)
            updated = torch.stack([values[assignment == j].mean(0)
                                   if (assignment == j).any() else centers[j]
                                   for j in range(len(centers))])
            if torch.allclose(centers, updated, atol=1e-6, rtol=0):
                centers = updated
                break
            centers = updated
        assignment = torch.cdist(values, centers).argmin(-1)
        category_counts = torch.bincount(assignment, minlength=len(centers))
        populated = category_counts > 0
        centers, category_counts = centers[populated], category_counts[populated]
        order = category_counts.argsort(descending=True, stable=True)
        length = len(order)
        priors[category, :length] = centers[order].exp()
        counts[category, :length] = category_counts[order]
        valid[category, :length] = True
    return dict(source_split='train', classes=list(classes),
                metric_size_priors=priors, valid_modes=valid, mode_counts=counts,
                seed=seed, max_modes=max_modes,
                min_samples_per_mode=min_samples_per_mode)
