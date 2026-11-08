"""Deterministic log-space size clustering, using training boxes only."""
import torch


def build_size_priors(sizes, labels, classes, num_modes=4,
                      seed=0, missing_class_size=None):
    """Fit a fixed K per category; preserve K even with scarce observations.

    When K exceeds the distinct supported modes, empty slots repeat populated
    centers and carry zero observed counts plus a replicated-prior flag.
    Completely absent categories use K explicit fixed placeholders instead.
    """
    sizes = torch.as_tensor(sizes, dtype=torch.float32, device='cpu')
    labels = torch.as_tensor(labels, dtype=torch.long, device='cpu')
    if sizes.ndim != 2 or sizes.shape[-1] != 3 or labels.shape != sizes.shape[:1]:
        raise ValueError('expected sizes [N, 3] and labels [N]')
    if not isinstance(num_modes, int) or isinstance(num_modes, bool) or num_modes < 1:
        raise ValueError('num_modes must be a positive integer')
    if not torch.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError('training size priors require finite positive box sizes')
    if ((labels < 0) | (labels >= len(classes))).any():
        raise ValueError('size-prior label is outside the class mapping')
    priors = torch.ones(len(classes), num_modes, 3)
    valid = torch.zeros(len(classes), num_modes, dtype=torch.bool)
    counts = torch.zeros(len(classes), num_modes, dtype=torch.long)
    fixed_prior_mask = torch.zeros_like(valid)
    replicated_prior_mask = torch.zeros_like(valid)
    fallback_size = None
    if missing_class_size is not None:
        fallback_size = torch.as_tensor(missing_class_size, dtype=torch.float32)
        if fallback_size.shape != (3,) or not torch.isfinite(fallback_size).all() or (fallback_size <= 0).any():
            raise ValueError('missing_class_size must contain three finite positive metric dimensions')
    generator = torch.Generator().manual_seed(seed)
    for category in range(len(classes)):
        values = sizes[labels == category].log()
        if len(values) == 0:
            if fallback_size is None:
                raise ValueError(f'No training boxes for category {classes[category]!r}')
            priors[category] = fallback_size
            valid[category] = True
            fixed_prior_mask[category] = True
            continue
        unique = values.unique(dim=0)
        centers = [unique[torch.randint(len(unique), (), generator=generator)]]
        for _ in range(1, num_modes):
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
        # K slots remain valid. Repeat supported centers for zero-count slots,
        # without duplicating observations or inventing cluster sample counts.
        empty_indices = torch.nonzero(~populated).flatten()
        populated_centers = centers[populated]
        if len(empty_indices):
            centers[empty_indices] = populated_centers[
                torch.arange(len(empty_indices)) % len(populated_centers)]
        order = category_counts.argsort(descending=True, stable=True)
        length = len(order)
        priors[category, :length] = centers[order].exp()
        counts[category, :length] = category_counts[order]
        valid[category, :length] = True
        replicated_prior_mask[category] = category_counts[order] == 0
    return dict(source_split='train', classes=list(classes),
                metric_size_priors=priors, valid_modes=valid, mode_counts=counts,
                fixed_prior_mask=fixed_prior_mask,
                replicated_prior_mask=replicated_prior_mask,
                missing_class_size=(fallback_size.tolist() if fallback_size is not None else None),
                seed=seed, num_modes=num_modes, clustering_strategy='fixed_k_log_kmeans')
