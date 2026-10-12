"""Fit a small baseline-change gate with run-held-out regional diagnostics.

Pure aqueous regions and explicit negatives fit the discriminant. Mixed and
presence-only regions constrain its threshold without becoming pixel labels.
"""
from dataclasses import asdict

import numpy as np

from .. import phase_reviewed_reference as rr
from .logistic import fit_weighted_logistic


def target(phase):
    if phase == 'aqueous':
        return 'pure_positive'
    if phase in ('water', 'co2_absent_region', 'aqueous_absent_region', 'current_experiment_co2_artifact'):
        return 'negative'
    if phase in ('mixed_gas_aqueous', 'mixed_aqueous_water', 'mixed_gas_aqueous_water',
                 'aqueous_present_region', 'aqueous_present_gas_absent_region'):
        return 'regional_positive'
    return None


def _fit(rows, backend='cpu'):
    supervised = [r for r in rows if r['kind'] in ('pure_positive', 'negative')]
    if any(sum(r['kind'] == k for r in supervised) < 4 for k in ('pure_positive', 'negative')):
        raise ValueError('Need at least four pure aqueous and four negative regions in the training split')
    strata = {(r['run'], r['kind']) for r in supervised}
    totals = {s: sum(r['weight'] for r in supervised if (r['run'], r['kind']) == s) for s in strata}
    x, y, w = [], [], []
    for r in supervised:
        values = np.asarray(r['features'], np.float32)
        x.append(values)
        y.extend([int(r['kind'] == 'pure_positive')]*len(values))
        w.extend([r['weight']/totals[r['run'], r['kind']]/len(values)]*len(values))
    fit = fit_weighted_logistic(np.concatenate(x), np.array(y), np.array(w),
                               regularization=.03, backend=backend, max_iter=80)
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(fit).items()}


def _regional_score(row, fit):
    q = .10 if row['kind'] == 'pure_positive' else .90
    return float(np.quantile(rr.scores(row['features'], {'fit': fit}), q))


def _rates(rows, threshold):
    strata = {}
    for r in rows:
        key = (r['run'], r['kind'] == 'negative')
        strata.setdefault(key, []).append(float(r['score'] >= threshold))
    negative = [np.mean(v) for (_, neg), v in strata.items() if neg]
    positive = [np.mean(v) for (_, neg), v in strata.items() if not neg]
    return dict(negative_region_acceptance=float(np.mean(negative)) if negative else None,
                positive_region_detection=float(np.mean(positive)) if positive else None)


def calibrate(rows, series, *, backend='cpu', maximum_negative_acceptance=.05,
              minimum_positive_detection=.70):
    """Group by complete run, never split pixels/adjacent boxes randomly.

    A single-run series can only supply temporal-block development checks, with
    adjacent reviewed frames purged. It is never an independent run holdout.
    """
    if not 0 <= maximum_negative_acceptance < 1 or not 0 < minimum_positive_detection <= 1:
        raise ValueError('Invalid acceptance policy')
    runs = sorted({r['run'] for r in rows})
    scored, folds = [], []
    if len(runs) > 1:
        splits = [(run, [r for r in rows if r['run'] != run], [r for r in rows if r['run'] == run]) for run in runs]
        scope = 'leave-one-run-out reference-gate development; optical parent has seen these runs'
    else:
        frames = sorted({r['frame'] for r in rows})
        splits = []
        for i, block in enumerate(np.array_split(frames, 4)):
            if not len(block):
                continue
            lo, hi = int(block[0]), int(block[-1])
            splits.append((f'time-block-{i}', [r for r in rows if r['frame'] < lo-1 or r['frame'] > hi+1],
                           [r for r in rows if lo <= r['frame'] <= hi]))
        scope = 'single-run purged temporal blocks; not independent spatial or run validation'
    for name, train, test in splits:
        try:
            fit = _fit(train, backend)
        except ValueError as error:
            folds.append(dict(group=name, status='insufficient_supervision', reason=str(error), held_out=len(test)))
            continue
        fold_rows = [dict(region_id=r['region_id'], run=r['run'], frame=r['frame'], kind=r['kind'],
                          score=_regional_score(r, fit)) for r in test]
        scored.extend(fold_rows)
        folds.append(dict(group=name, status='scored', training_regions=len(train), held_out=len(test)))
    fit = _fit(rows, backend)
    # Runtime scores are float32. A float64 nextafter can round back onto the
    # rejected score at inference and silently reverse a threshold decision.
    candidates = np.unique([.5, 1., *[float(np.nextafter(np.float32(r['score']), np.float32(1.))) for r in scored]])
    feasible = []
    for threshold in candidates:
        rates = _rates(scored, threshold)
        if rates['negative_region_acceptance'] is not None and rates['positive_region_detection'] is not None:
            if rates['negative_region_acceptance'] <= maximum_negative_acceptance:
                feasible.append((rates['positive_region_detection'], -rates['negative_region_acceptance'], float(threshold), rates))
    best = max(feasible, key=lambda x: x[:3]) if feasible else (0., 0., 1., _rates([], 1.))
    complete = len(scored) == len(rows)
    enabled = complete and best[0] >= minimum_positive_detection
    calibration = dict(feature_version=rr.FEATURE_VERSION, series=series, fit=fit,
        threshold=best[2], enabled=enabled, policy=dict(maximum_negative_region_acceptance=maximum_negative_acceptance,
        minimum_positive_region_detection=minimum_positive_detection),
        scope=scope, independent_validation=False, split_complete=complete,
        metrics=best[3], reason='passed grouped development policy' if enabled else 'insufficient separation or incomplete grouped validation; C2 additions disabled')
    return calibration, dict(folds=folds, out_of_fold_regions=scored, calibration=calibration,
        counts={k: sum(r['kind'] == k for r in rows) for k in ('negative', 'pure_positive', 'regional_positive')},
        pixel_labels_used_for_mixed_boxes=False, random_pixel_split=False)
