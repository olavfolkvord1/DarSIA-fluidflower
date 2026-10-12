"""Bounded threshold selection on immutable, C2-supported review evidence.

Whole-region labels are never converted to dense positive targets. Negative
coverage is evaluated per region and per run, not pooled over a dominant run.
These are development checks on selected reviews, not independent accuracy.
"""
import numpy as np


THRESHOLDS = tuple(float(np.float32(t)) for t in (.05, .075, .10, .15, .20, .30, .40, .60, 1.))


def negative_coverage(row, threshold):
    scores = np.asarray(row['proposal_scores'], np.float32)
    counts = np.asarray(row.get('proposal_counts', np.ones(len(scores))), dtype=np.float64)
    return float(np.sum(counts[scores >= threshold])/counts.sum() * row['proposal_fraction']) if len(scores) else 0.


def evaluate(rows, threshold):
    negative = [r for r in rows if r['kind'] == 'negative']
    rates = [negative_coverage(r, threshold) for r in negative]
    by_run = {run: float(np.mean([negative_coverage(r, threshold) for r in negative if r['run'] == run]))
              for run in sorted({r['run'] for r in negative})}
    positives = [r for r in rows if r['kind'] != 'negative' and len(r['proposal_scores'])]
    detected = [float(negative_coverage({**r, 'proposal_fraction': 1.}, threshold) >= .1) for r in positives]
    return dict(maximum_negative_box_addition=max(rates, default=0.), negative_addition_by_run=by_run,
                positive_region_signal_detected=float(np.mean(detected)) if detected else None,
                negative_regions=len(negative), negative_regions_with_proposals=sum(bool(r['proposal_scores']) for r in negative),
                positive_regions=len(positives))


def select_threshold(rows, *, fallback, maximum_box_addition=.01, minimum_negative_regions=4,
                     minimum_positive_regions=8, minimum_runs=2):
    negative = [r for r in rows if r['kind'] == 'negative']
    positive = [r for r in rows if r['kind'] != 'negative' and len(r['proposal_scores'])]
    supported = (len(negative) >= minimum_negative_regions and len(positive) >= minimum_positive_regions
                 and len({r['run'] for r in negative}) >= minimum_runs
                 and len({r['run'] for r in positive}) >= minimum_runs)
    # A sparse facies may tighten a demonstrably unsafe fallback, but cannot
    # become more permissive just because negative controls are missing.
    candidates = [t for t in THRESHOLDS if supported or t >= fallback]
    candidates = sorted(set([float(fallback), *candidates])) if not supported else candidates
    feasible = [t for t in candidates if evaluate(rows, t)['maximum_negative_box_addition'] <= maximum_box_addition]
    value = min(feasible) if feasible else 1.
    return dict(threshold=value, sufficient_facies_evidence=supported,
                reason='bounded negative-control selection' if supported else 'sparse controls: no relaxation below fallback',
                metrics=evaluate(rows, value), fallback_metrics=evaluate(rows, fallback))


def calibrate(rows, *, fallback=.10, maximum_box_addition=.01):
    if not rows or any(r.get('kind') not in ('negative', 'pure_positive', 'regional_positive') for r in rows):
        raise ValueError('Typed facies review evidence is required')
    for row in rows:
        scores = np.asarray(row['proposal_scores'])
        counts = np.asarray(row.get('proposal_counts', np.ones(len(scores))))
        if (scores.ndim != 1 or counts.shape != scores.shape or not np.isfinite(scores).all()
                or np.any((scores < 0) | (scores > 1)) or not np.isfinite(counts).all() or np.any(counts <= 0)
                or not 0 <= row['proposal_fraction'] <= 1):
            raise ValueError('Invalid frozen proposal scores or counts')
    fallback = float(np.float32(fallback))
    results = {}
    for facies in sorted({r['facies'] for r in rows}):
        group = [r for r in rows if r['facies'] == facies]
        selected = select_threshold(group, fallback=fallback, maximum_box_addition=maximum_box_addition)
        folds = []
        for run in sorted({r['run'] for r in group}):
            train, test = [r for r in group if r['run'] != run], [r for r in group if r['run'] == run]
            trial = select_threshold(train, fallback=fallback, maximum_box_addition=maximum_box_addition,
                                     minimum_runs=1)
            metrics = evaluate(test, trial['threshold'])
            folds.append(dict(excluded_run=run, threshold=trial['threshold'], metrics=metrics))
        selected['run_excluded_development_checks'] = folds
        if selected['threshold'] < fallback and any(
                f['metrics']['maximum_negative_box_addition'] > maximum_box_addition for f in folds):
            selected.update(threshold=fallback, reason='run-excluded negative check failed: retained fallback',
                            metrics=evaluate(group, fallback))
        results[str(facies)] = selected
    return dict(threshold=fallback, facies_thresholds={k: v['threshold'] for k, v in results.items()},
                facies=results, maximum_new_c2_pixels_per_negative_box=maximum_box_addition,
                scope='threshold development on reviewed regions; frozen optical fit has seen these runs',
                independent_validation=False, positive_boxes_used_as_dense_labels=False,
                acceptance_scores_are_not_probabilities=True)
