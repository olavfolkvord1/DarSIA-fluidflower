"""Explicit geometry-only extension of frozen Rule B classes.

The inherited classes remain exact. Newly accepted rig pixels inherit the
nearest frozen barrier/open-component class, without relabelling components,
learning a boundary, or using any predicted phase category.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.ndimage import distance_transform_edt

CONTRACT = "ffac-frozen-barrier-geometric-extension-v1"


def array_sha256(array):
    array=np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def extend_frozen_barrier_components(open_components,barrier_mask,rig_domain):
    """Extend the nearest frozen class only into explicitly accepted rig pixels.

    Inputs already occupy the declared phase grid. Any reference transport is
    separately source-bound by the caller; this function performs no warp.
    """
    components=np.asarray(open_components);barrier=np.asarray(barrier_mask);domain=np.asarray(rig_domain)
    if components.ndim!=2 or components.dtype.kind not in 'iu' or np.any(components<0):
        raise TypeError("Frozen open components require a nonnegative integer grid")
    if barrier.dtype!=bool or domain.dtype!=bool or barrier.shape!=components.shape or domain.shape!=components.shape:
        raise TypeError("Frozen barrier and accepted rig domain require the exact same boolean phase grid")
    if np.any(barrier & (components>0)):
        raise ValueError("Frozen pixels cannot be both a barrier and an open component")
    known=barrier | (components>0)
    extension=domain & ~known
    if extension.any() and not known.any():
        raise ValueError("An accepted rig cannot inherit classes from an empty frozen reference")
    extended_components=components.copy();extended_barrier=barrier.copy()
    if extension.any():
        nearest=distance_transform_edt(~known,return_distances=False,return_indices=True)
        rows,cols=nearest[:,extension]
        extended_components[extension]=components[rows,cols]
        extended_barrier[extension]=barrier[rows,cols]
    if not np.array_equal(extended_components[known],components[known]) or not np.array_equal(extended_barrier[known],barrier[known]):
        raise ValueError("Geometric extension changed an inherited frozen class")
    if np.any(domain & ~(extended_barrier | (extended_components>0))):
        raise ValueError("Accepted rig contains unclassified Rule B geometry")
    audit=dict(contract=CONTRACT,assumption="nearest frozen static barrier or same open-component ID",
        method="scipy Euclidean nearest index; deterministic library tie convention",
        source_component_ids_unchanged=True,reference_classes_preserved=True,
        phase_shape_hw=list(components.shape),known_pixels=int(known.sum()),
        extended_pixels=int(extension.sum()),extended_barrier_pixels=int((extension&extended_barrier).sum()),
        extended_open_pixels=int((extension&(extended_components>0)).sum()),
        inherited_open_ids=[int(value) for value in np.unique(components) if value>0],
        frozen_open_components_sha256=array_sha256(components),frozen_barrier_sha256=array_sha256(barrier),
        known_mask_sha256=array_sha256(known),extension_mask_sha256=array_sha256(extension),
        phase_categories_used=False,components_relabelled=False)
    return dict(open_components=extended_components,barrier_mask=extended_barrier,
                known=known,geometric_extension=extension,reference_domain=domain.copy()),audit
