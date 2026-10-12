"""Semantics-preserving TPE acceleration for FF-AC's grouped search space."""

from __future__ import annotations

import atexit
import copy
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import distribution_to_json
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import _split_trials
from optuna.trial import FrozenTrial, TrialState


SAMPLER_CONTRACT = "optuna-4.9-group-split-cache-v1"
DEFAULT_TPE_THREADS = 4

_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_WORKERS = 0
_EXECUTOR_LOCK = threading.Lock()


def _configured_tpe_threads() -> int:
    raw = os.environ.get("FFAC_TPE_THREADS", str(DEFAULT_TPE_THREADS)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_TPE_THREADS


def _shared_executor() -> ThreadPoolExecutor | None:
    """Return one process-wide executor so sequential jobs do not leak threads."""

    global _EXECUTOR, _EXECUTOR_WORKERS
    workers = _configured_tpe_threads()
    if workers <= 1:
        return None
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="ffac-tpe",
            )
            _EXECUTOR_WORKERS = workers
        elif _EXECUTOR_WORKERS != workers:
            _EXECUTOR.shutdown(wait=True)
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="ffac-tpe",
            )
            _EXECUTOR_WORKERS = workers
        return _EXECUTOR


def _shutdown_executor() -> None:
    global _EXECUTOR, _EXECUTOR_WORKERS
    with _EXECUTOR_LOCK:
        if _EXECUTOR is not None:
            _EXECUTOR.shutdown(wait=True)
            _EXECUTOR = None
            _EXECUTOR_WORKERS = 0


atexit.register(_shutdown_executor)


class CachedGroupedTPESampler(TPESampler):
    """Avoid repeated history scans inside one grouped TPE proposal.

    Optuna 4.9 calls ``_split_trials`` once for every decomposed parameter
    group.  The split depends only on trial values and constraints, not on the
    group's distributions.  FF-AC currently has eight groups, so calculating it
    once preserves the generated samples while removing repeated O(history)
    work.  The accumulated group decomposition is also updated only with trials
    not seen by this sampler instance.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ffac_group_study_id: int | None = None
        self._ffac_group_seen_trial_ids: set[int] = set()
        self._ffac_observation_study_id: int | None = None
        self._ffac_observation_rows: dict[int, int] = {}
        self._ffac_observation_values: dict[str, list[float]] = {}
        self._ffac_observation_present: dict[str, list[bool]] = {}
        self._ffac_observation_distributions: dict[str, str] = {}

    def _reset_observation_cache(self, study_id: int) -> None:
        self._ffac_observation_study_id = study_id
        self._ffac_observation_rows.clear()
        self._ffac_observation_values.clear()
        self._ffac_observation_present.clear()
        self._ffac_observation_distributions.clear()

    def _sync_observation_cache(
        self,
        study: optuna.Study,
        trials: list[FrozenTrial],
        search_space: dict[str, BaseDistribution],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Cache parameter conversion while retaining Optuna's trial ordering."""

        study_id = int(study._study_id)
        fingerprints = {
            name: json.dumps(
                json.loads(distribution_to_json(distribution)),
                sort_keys=True,
                separators=(",", ":"),
            )
            for name, distribution in search_space.items()
        }
        if self._ffac_observation_study_id != study_id:
            self._reset_observation_cache(study_id)
        elif (
            self._ffac_observation_distributions
            and self._ffac_observation_distributions != fingerprints
        ):
            # Dynamic search spaces are valid. Rebuild if a parameter appears or
            # its distribution changes so no stale internal representation leaks.
            self._reset_observation_cache(study_id)
        self._ffac_observation_distributions = fingerprints
        for name in search_space:
            self._ffac_observation_values.setdefault(name, [])
            self._ffac_observation_present.setdefault(name, [])

        for trial in trials:
            trial_id = int(trial._trial_id)
            if trial_id in self._ffac_observation_rows:
                continue
            params = self._get_params(trial)
            self._ffac_observation_rows[trial_id] = len(
                self._ffac_observation_rows
            )
            for name, distribution in search_space.items():
                present = name in params
                self._ffac_observation_present[name].append(present)
                self._ffac_observation_values[name].append(
                    distribution.to_internal_repr(params[name])
                    if present
                    else np.nan
                )

        values = {
            name: np.asarray(items)
            for name, items in self._ffac_observation_values.items()
        }
        present = {
            name: np.asarray(items, dtype=bool)
            for name, items in self._ffac_observation_present.items()
        }
        return values, present

    def _trial_indices(self, trials: list[FrozenTrial]) -> np.ndarray:
        return np.fromiter(
            (
                self._ffac_observation_rows[int(trial._trial_id)]
                for trial in trials
            ),
            dtype=np.int64,
            count=len(trials),
        )

    def _build_cached_parzen_estimator(
        self,
        search_space: dict[str, BaseDistribution],
        trial_indices: np.ndarray,
        values: dict[str, np.ndarray],
        present: dict[str, np.ndarray],
    ) -> Any:
        mask = np.ones(len(trial_indices), dtype=bool)
        for name in search_space:
            mask &= present[name][trial_indices]
        selected = trial_indices[mask]
        observations = {
            name: values[name][selected] for name in search_space
        }
        return self._parzen_estimator_cls(
            observations,
            search_space,
            self._parzen_estimator_parameters,
        )

    def infer_relative_search_space(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        if not self._multivariate or not self._group:
            return super().infer_relative_search_space(study, trial)
        if self._group_decomposed_search_space is None:
            raise RuntimeError("Optuna grouped search-space helper is unavailable")
        helper = self._group_decomposed_search_space
        if self._ffac_group_study_id is None:
            self._ffac_group_study_id = study._study_id
        elif self._ffac_group_study_id != study._study_id:
            raise ValueError("CachedGroupedTPESampler cannot serve multiple studies")

        # JournalStorage returns a cached trial list.  Iterating it is cheap; the
        # expensive repeated add_distributions calls are limited to new trials.
        trials = study._get_trials(
            deepcopy=False,
            states=(TrialState.COMPLETE,),
            use_cache=True,
        )
        search_space_group = helper._search_space
        for prior in trials:
            trial_id = int(prior._trial_id)
            if trial_id in self._ffac_group_seen_trial_ids:
                continue
            search_space_group.add_distributions(prior.distributions)
            self._ffac_group_seen_trial_ids.add(trial_id)
        self._search_space_group = copy.deepcopy(search_space_group)

        search_space: dict[str, BaseDistribution] = {}
        for sub_space in self._search_space_group.search_spaces:
            for name, distribution in sorted(sub_space.items()):
                if not distribution.single():
                    search_space[name] = distribution
        return search_space

    def sample_relative(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if not self._group or self._constant_liar:
            return super().sample_relative(study, trial, search_space)
        if self._search_space_group is None:
            return {}

        trials = study._get_trials(
            deepcopy=False,
            states=(TrialState.COMPLETE, TrialState.PRUNED),
            use_cache=True,
        )
        if len(trials) < self._n_startup_trials:
            return {}
        count = sum(prior.state != TrialState.RUNNING for prior in trials)
        below_trials, above_trials = _split_trials(
            study,
            trials,
            self._gamma(count),
            self._constraints_func is not None,
        )

        active_spaces: list[dict[str, BaseDistribution]] = []
        for sub_space in self._search_space_group.search_spaces:
            active: dict[str, BaseDistribution] = {}
            for name, distribution in sorted(sub_space.items()):
                if distribution.single() or name not in search_space:
                    continue
                active[name] = distribution
            if not active:
                continue
            active_spaces.append(active)

        # The optimized array path intentionally targets FF-AC's single-objective,
        # non-constant-liar studies. Fall back to Optuna for other contracts.
        if study._is_multi_objective():
            return super().sample_relative(study, trial, search_space)
        values, present = self._sync_observation_cache(
            study,
            trials,
            search_space,
        )
        below_indices = self._trial_indices(below_trials)
        above_indices = self._trial_indices(above_trials)
        estimators = [
            (
                active,
                self._build_cached_parzen_estimator(
                    active,
                    below_indices,
                    values,
                    present,
                ),
                self._build_cached_parzen_estimator(
                    active,
                    above_indices,
                    values,
                    present,
                ),
            )
            for active in active_spaces
        ]

        # Sampling remains serial and in the original group order. This preserves
        # the RNG stream exactly; only the independent likelihood calculations run
        # concurrently.
        samples = [
            below.sample(self._rng.rng, self._n_ei_candidates)
            for _active, below, _above in estimators
        ]

        def _acquisition(item: tuple[Any, dict[str, np.ndarray]]) -> np.ndarray:
            (_active, below, above), sampled = item
            return self._compute_acquisition_func(sampled, below, above)

        executor = _shared_executor()
        if executor is None or len(estimators) <= 1:
            acquisitions = [
                _acquisition(item) for item in zip(estimators, samples)
            ]
        else:
            acquisitions = list(executor.map(_acquisition, zip(estimators, samples)))

        params: dict[str, Any] = {}
        for (active, _below, _above), sampled, acquisition in zip(
            estimators,
            samples,
            acquisitions,
        ):
            selected = TPESampler._compare(sampled, acquisition)
            for name, distribution in active.items():
                params[name] = distribution.to_external_repr(selected[name])
        return params
