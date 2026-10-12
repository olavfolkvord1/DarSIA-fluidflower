"""Bound overlapping optical/phase work within an existing Studio budget."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics


CALIBRATION_FORMAT = 'phase-throughput-calibration-v1'
COMBINED_SCOPES = ('CPU phase + PNG + proposals', 'CPU phase + PNG; proposals deferred')


def calibration_identity():
    """Hardware and execution identity; learned weights are deliberately absent."""
    import cv2
    import numpy
    import scipy
    import psutil
    from darsia.fluidflower.tools import phase_regional_pipeline as pipeline
    from darsia.fluidflower.tools import phase_regional_fused_execution as fused
    from darsia.fluidflower import phase_learning, phase_validation, phase_execution_resources
    modules = [pipeline, fused, *pipeline.REVIEWED_SOURCE_SHA256, fused.rendering,
               phase_learning, phase_validation, phase_execution_resources]
    paths = {Path(module.__file__).name: Path(module.__file__) for module in modules}
    paths[Path(__file__).name] = Path(__file__)
    transport = Path(phase_validation.__file__).with_name('phase_proposal_transport.py')
    if transport.exists():
        paths[transport.name] = transport
    return dict(
        hardware=dict(system=platform.system(), machine=platform.machine(),
            processor=platform.processor(), logical_cpus=psutil.cpu_count(logical=True),
            physical_cpus=psutil.cpu_count(logical=False),
            ram_class_gib=int(round(psutil.virtual_memory().total / (8 * 1024**3))) * 8),
        libraries=dict(numpy=numpy.__version__, scipy=scipy.__version__, cv2=cv2.__version__),
        sources={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()})


def _valid_hashes(value, *, allow_empty=False):
    return (isinstance(value, dict) and (bool(value) or allow_empty)
            and all(isinstance(key, str) and isinstance(digest, str) and len(digest) == 64
                    and all(char in '0123456789abcdef' for char in digest)
                    for key, digest in value.items()))


def build_calibration(receipts, frame_pixels, *, identity=None, native_threads=None):
    """Build a portable JSON record from verified combined-pipeline receipts.

    Accepts combined-pipeline rows or the complete CPU-stage benchmark document
    (concurrency rows with verified score, diagnostic and report equality). CPU
    rows alone require native_threads explicitly. Repeated measurements are
    grouped by worker count. The output/proposal workload and task count must
    remain constant. Callers publish the record in the series config.
    """
    scope, isolated = 'CPU phase + PNG + proposals', None
    generate_review_proposals = True
    document = isinstance(receipts, dict)
    expected_identity = identity or calibration_identity()
    if document:
        generate_review_proposals = receipts.get('generate_review_proposals', True)
        measured_identity = receipts.get('execution_identity')
        if measured_identity != expected_identity:
            raise ValueError('Benchmark execution identity is missing or differs from the current runtime')
        if receipts.get('execution_identity_after', measured_identity) != measured_identity:
            raise ValueError('Benchmark execution identity changed during measurement')
        native_threads = receipts.get('threads_per_worker', native_threads)
        isolated = receipts.get('production_isolated')
        if 'concurrency' in receipts:
            if receipts.get('worker_results_exact_match') is not True:
                raise ValueError('CPU benchmark has no verified worker output comparison')
            receipts = receipts['concurrency']
        elif 'configurations' in receipts:
            if receipts.get('all_configurations_exact_first_frame_match') is not True:
                raise ValueError('Combined benchmark has no verified worker output comparison')
            receipts = receipts['configurations']
        else:
            raise ValueError('Unknown benchmark receipt document')
    receipts = list(receipts)
    if not document:
        if identity is None and any(row.get('execution_identity') != expected_identity for row in receipts):
            raise ValueError('Standalone receipts require their measured execution identity')
        measured_identity = expected_identity
        isolated = bool(receipts) and all(row.get('production_isolated') is True for row in receipts)
    if isolated is not True or any(row.get('production_isolated', True) is not True for row in receipts):
        raise ValueError('Automatic calibration requires production-isolated benchmark measurements')
    if receipts and 'workers' in receipts[0]:
        scope = 'CPU phase only; encoding, proposals and optical inference excluded'
        normalized = []
        for row in receipts:
            if row.get('verified') is not True or not _valid_hashes(row.get('hashes')):
                raise ValueError('CPU benchmark row is not output-verified')
            if not _valid_hashes(row.get('diagnostics'), allow_empty=True) or not isinstance(row.get('report'), dict):
                raise ValueError('CPU benchmark must verify diagnostics and scientific report')
            hashes = {f'score/{key}': value for key, value in row['hashes'].items()}
            hashes.update({f'diagnostic/{key}': value for key, value in row['diagnostics'].items()})
            hashes['report'] = hashlib.sha256(json.dumps(row['report'], sort_keys=True,
                separators=(',', ':')).encode()).hexdigest()
            normalized.append(dict(phase_workers=row['workers'], tasks=row['tasks'],
                wall_s=row['wall_seconds'], phase_threads=row.get('phase_threads', native_threads),
                first_frame_hashes=hashes, output_workers=0, proposal_workers=0, output_pending=0,
                generate_review_proposals=False))
        receipts = normalized
    else:
        receipts = [{**row, 'generate_review_proposals': row.get('generate_review_proposals', generate_review_proposals)}
                    for row in receipts]
        if receipts and receipts[0]['generate_review_proposals'] is False:
            scope = 'CPU phase + PNG; proposals deferred'
    if not receipts or int(frame_pixels) <= 0:
        raise ValueError('Calibration requires measurements and a positive image size')
    reference = receipts[0].get('first_frame_hashes')
    if not _valid_hashes(reference):
        raise ValueError('Calibration requires explicit verified output SHA-256 values')
    workload_keys = ('output_workers', 'proposal_workers', 'output_pending', 'tasks', 'generate_review_proposals')
    workload = {key: receipts[0].get(key) for key in workload_keys}
    if any(value is None for value in workload.values()):
        raise ValueError('Calibration receipt is missing the fixed output workload')
    if not isinstance(workload['generate_review_proposals'], bool):
        raise ValueError('Proposal generation mode must be an explicit boolean')
    if not workload['generate_review_proposals'] and workload['proposal_workers']:
        raise ValueError('Deferred proposal generation cannot reserve proposal workers')
    grouped = {}
    for row in receipts:
        if row.get('phase_threads') != 1 or row.get('first_frame_hashes') != reference:
            raise ValueError('Calibration requires one native thread and identical output hashes')
        if any(row.get(key) != value for key, value in workload.items()):
            raise ValueError('Calibration must compare the same output/proposal workload')
        workers, tasks = int(row['phase_workers']), int(row['tasks'])
        seconds = float(row['wall_s'])
        if workers < 1 or tasks < 1 or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Calibration measurements must be positive and finite')
        grouped.setdefault(workers, []).append(seconds / tasks)
    return dict(format=CALIBRATION_FORMAT, identity=deepcopy(measured_identity),
        frame_pixels=int(frame_pixels), native_threads=1, workload=workload,
        measurement_scope=scope, production_isolated=isolated,
        reference_output_sha256=reference,
        measurements=[dict(workers=workers, seconds_per_frame=samples,
                           exact_verified=True, output_sha256=reference)
                      for workers, samples in sorted(grouped.items())])


def calibrated_workers(path, pixels, worker_limit, *, identity=None, workload=None,
                       generate_review_proposals=None):
    """Fail closed on stale or incomplete calibration; never select an untested count."""
    if path is None:
        return None, 'no calibration configured'
    try:
        payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if payload.get('format') != CALIBRATION_FORMAT:
            return None, 'unrecognized calibration format'
        if payload.get('production_isolated') is not True:
            return None, 'calibration was not measured with production isolated'
        if payload.get('identity') != (identity or calibration_identity()):
            return None, 'hardware or execution identity changed'
        native_pixels = int(payload.get('frame_pixels', 0))
        if native_pixels <= 0 or not .8 <= pixels / native_pixels <= 1.2:
            return None, 'image size outside calibrated 20 percent band'
        if payload.get('native_threads') != 1:
            return None, 'calibration requires one native thread'
        if payload.get('measurement_scope') in COMBINED_SCOPES:
            measured_workload = dict(payload.get('workload', {}))
            measured_workload.setdefault('generate_review_proposals', True)
            if (generate_review_proposals is not None and
                    measured_workload['generate_review_proposals'] is not generate_review_proposals):
                return None, 'calibrated proposal generation mode differs from the requested workload'
        if workload is not None and payload.get('measurement_scope') in COMBINED_SCOPES:
            if any(measured_workload.get(key) != value for key, value in workload.items()):
                return None, 'calibrated output workload differs from the actual allocation'
        reference = payload.get('reference_output_sha256')
        if not _valid_hashes(reference):
            return None, 'calibration has no verified reference hashes'
        candidates = {}
        for row in payload.get('measurements', []):
            workers = int(row.get('workers', 0))
            if not 1 <= workers <= worker_limit:
                continue
            if row.get('exact_verified') is not True or row.get('output_sha256') != reference:
                continue
            samples = row.get('seconds_per_frame')
            if not isinstance(samples, list) or not samples:
                continue
            samples = [float(value) for value in samples]
            if any(not math.isfinite(value) or value <= 0 for value in samples):
                continue
            candidates.setdefault(workers, []).extend(samples)
        if not candidates:
            return None, 'no verified calibration entries fit the current resource budget'
        medians = {workers: statistics.median(samples) for workers, samples in candidates.items()}
        fastest = min(medians.values())
        workers = min(workers for workers, median in medians.items() if median <= fastest * 1.05)
        return workers, 'smallest verified count within 5 percent of fastest measured median'
    except (OSError, ValueError, TypeError, AttributeError, OverflowError) as error:
        return None, f'calibration unavailable: {type(error).__name__}'


def available_logical_cpus():
    """Usable logical CPUs, including the current process affinity limit."""
    count = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    try:
        import psutil
    except ImportError:
        return max(1, int(count))
    try:
        logical = psutil.cpu_count(logical=True) or count
        count = min(count, logical)
        affinity = getattr(psutil.Process(), 'cpu_affinity', None)
        if affinity is not None:
            count = min(count, len(affinity()))
    except (OSError, AttributeError, psutil.Error):
        pass
    return max(1, int(count))


def total_cpu_recommendation(default_cpus, calibration_path, *, reserve=None, generate_review_proposals=True):
    """Expand a conservative default only from verified local measurements.

    Callers must preserve explicit --cpus / PHASE_CPU_LIMIT overrides instead
    of passing those user-selected values here. The reserve covers finalizer,
    PNG writers, proposal helpers, optical parity checks and the coordinator.
    The image is not yet loaded by launchers: the calibration's own valid image
    size is used here; throughput_resources checks the actual image-size band
    before enabling its calibrated phase-worker count.
    """
    default_cpus = int(default_cpus)
    explicit_reserve = reserve is not None
    reserve = int(reserve) if explicit_reserve else (10 if generate_review_proposals else 8)
    if default_cpus < 1 or reserve < 0:
        raise ValueError('Positive default CPU budget and nonnegative reserve required')
    available_limit = max(1, available_logical_cpus() - 1)
    result = dict(cpu_limit=default_cpus, default_cpus=default_cpus,
                  available_cpu_limit=available_limit, reserved_cpu_slots=reserve,
                  phase_workers=None, selection_source='default',
                  calibration_path=str(calibration_path) if calibration_path is not None else None,
                  reason='no calibration configured', generate_review_proposals=generate_review_proposals)
    if calibration_path is None:
        return result
    try:
        payload = json.loads(Path(calibration_path).read_text(encoding='utf-8-sig'))
        pixels = int(payload.get('frame_pixels', 0))
        if pixels <= 0:
            result['reason'] = 'calibration has no valid image size'
            return result
        if payload.get('measurement_scope') in COMBINED_SCOPES:
            workload = _output_workload(payload)
            if workload['generate_review_proposals'] is not generate_review_proposals:
                result['reason'] = 'calibrated proposal generation mode differs from the requested workload'
                return result
            if not explicit_reserve:
                reserve = 6 + workload['output_workers'] + workload['proposal_workers']
                result['reserved_cpu_slots'] = reserve
        selected, reason = calibrated_workers(calibration_path, pixels,
            max(0, available_limit - reserve), generate_review_proposals=generate_review_proposals)
        result['reason'] = reason
        if selected is None:
            return result
        result.update(cpu_limit=min(available_limit, max(default_cpus, selected + reserve)),
                      phase_workers=selected, selection_source='verified calibration')
        return result
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
        result['reason'] = f'calibration unavailable: {type(error).__name__}'
        return result


def recommended_total_cpus(default_cpus, calibration_path, *, reserve=None, generate_review_proposals=True):
    """Integer launcher convenience API; total_cpu_recommendation logs the decision."""
    return total_cpu_recommendation(default_cpus, calibration_path, reserve=reserve,
        generate_review_proposals=generate_review_proposals)['cpu_limit']


def _output_workload(payload):
    measured = payload.get('workload', {})
    workload = {name: int(measured[name]) for name in
                ('output_workers', 'proposal_workers', 'output_pending')}
    if any(measured[name] != value for name, value in workload.items()):
        raise ValueError('Calibration output resource counts must be integers')
    proposal_limit = max(0, available_logical_cpus() - 1)
    if workload['output_workers'] < 1 or not 0 <= workload['proposal_workers'] <= proposal_limit:
        raise ValueError('Unsupported calibrated output/proposal worker count')
    if workload['output_pending'] < workload['output_workers']:
        raise ValueError('Calibrated output queue must fit its workers')
    workload['generate_review_proposals'] = measured.get('generate_review_proposals', True)
    if not isinstance(workload['generate_review_proposals'], bool):
        raise ValueError('Proposal generation mode must be an explicit boolean')
    if not workload['generate_review_proposals'] and workload['proposal_workers']:
        raise ValueError('Deferred proposals cannot reserve proposal workers')
    return workload


def _overlap_memory(resources, pixels, *, generate_review_proposals=True):
    # In addition to the GPU-owner/queued-output accounting, proposal helpers
    # retain transported float32 fields and construct full-frame float64
    # entropy temporaries. Include those helpers in the shared memory budget.
    proposals = int(getattr(resources, 'proposal_workers', 0)) if generate_review_proposals else 0
    proposal_bytes = proposals * (pixels * 192 + 512 * 1024**2)
    shared = pixels * (64 + 24 * resources.pending_outputs) + 512 * 1024**2 + proposal_bytes
    per_frame = max(resources.estimated_working_bytes, pixels * 128)
    return shared, per_frame, proposal_bytes


def apply_calibrated_output_resources(resources, shape_hw, calibration_path, *, enabled=True,
                                     generate_review_proposals=True):
    """Return (allocation, receipt); unmatched or unaffordable plans keep defaults.

    The caller skips this for explicit CPU execution. An automatic CUDA-to-CPU
    fallback may retain the same bounded output helpers while phase overlap is
    disabled. A calibration's helper counts are applied exactly, never clamped
    to an unmeasured combination.
    """
    released = int(resources.proposal_workers) if not generate_review_proposals else 0
    if released:
        resources = replace(resources, proposal_workers=0,
                            compute_threads=resources.compute_threads + released)
    if enabled is None:
        enabled = os.environ.get('PHASE_THROUGHPUT', '1').lower() in ('1', 'true', 'yes', 'on')
    enabled = enabled and os.environ.get('PHASE_THROUGHPUT', '1').lower() in ('1', 'true', 'yes', 'on')
    decision = dict(applied=False, reason='disabled' if not enabled else 'no calibration configured',
                    calibration_path=str(calibration_path) if calibration_path is not None else None,
                    requested=None, actual=resources.to_dict(),
                    generate_review_proposals=generate_review_proposals, released_proposal_cpu_slots=released)
    if not enabled or calibration_path is None:
        return resources, decision
    try:
        payload = json.loads(Path(calibration_path).read_text(encoding='utf-8-sig'))
        if payload.get('measurement_scope') not in COMBINED_SCOPES:
            decision['reason'] = 'calibration did not measure output/proposal work'
            return resources, decision
        workload = _output_workload(payload)
        decision['requested'] = workload
        if workload['generate_review_proposals'] is not generate_review_proposals:
            decision['reason'] = 'calibrated proposal generation mode differs from the requested workload'
            return resources, decision
        compute = resources.cpu_limit - workload['output_workers'] - workload['proposal_workers']
        if compute < 6:
            decision['reason'] = 'CPU budget cannot fit calibrated helpers and phase overlap'
            return resources, decision
        candidate = replace(resources, compute_threads=compute,
            output_workers=workload['output_workers'], proposal_workers=workload['proposal_workers'],
            pending_outputs=workload['output_pending'])
        pixels = int(shape_hw[0]) * int(shape_hw[1])
        shared, per_frame, proposal_bytes = _overlap_memory(candidate, pixels,
            generate_review_proposals=generate_review_proposals)
        limit = min(compute - 4, max(0, int((candidate.memory_bytes - shared) // per_frame)))
        selected, reason = calibrated_workers(calibration_path, pixels, limit, workload=workload,
            generate_review_proposals=generate_review_proposals)
        if selected is None or selected < 2:
            decision['reason'] = reason if selected is None else 'calibration selected serial execution'
            return resources, decision
        decision.update(applied=True, reason=reason, phase_workers=selected,
                        estimated_bytes=shared + selected * per_frame,
                        proposal_estimated_bytes=proposal_bytes, actual=candidate.to_dict())
        return candidate, decision
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
        decision['reason'] = f'calibration unavailable: {type(error).__name__}'
        return resources, decision


@dataclass(frozen=True)
class ThroughputResources:
    enabled: bool
    phase_workers: int
    phase_threads: int
    optical_threads: int
    pending_frames: int
    estimated_bytes: int
    reason: str
    selection_source: str = 'conservative'
    requested_phase_workers: int | None = None
    worker_limit: int = 0
    calibration_path: str | None = None
    calibration_status: str | None = None
    proposal_estimated_bytes: int = 0
    generate_review_proposals: bool = True

    def to_dict(self):
        return asdict(self)


def throughput_resources(resources, shape_hw, *, enabled=None, calibration_path=None,
                         generate_review_proposals=True):
    """Reserve optical checks + coordinator; never multiply native CPU teams.

    This plan is activated only after CUDA selection. CPU tile inference changes
    process-global native thread limits and retains the established serial path.
    Estimates bound extra queued work, not the existing runtime's absolute RSS.
    """
    if enabled is None:
        enabled = os.environ.get('PHASE_THROUGHPUT', '1').lower() in ('1', 'true', 'yes', 'on')
    pixels = int(shape_hw[0]) * int(shape_hw[1])
    shared, per_frame, proposal_bytes = _overlap_memory(resources, pixels,
        generate_review_proposals=generate_review_proposals)
    compute_threads = resources.compute_threads + (int(getattr(resources, 'proposal_workers', 0))
                                                   if not generate_review_proposals else 0)
    optical = 2
    # Reserve up to two coordinator/reference threads while reviewed-frame
    # workers are active (the current native team uses only one of them).
    usable = compute_threads - optical - 2
    memory_slots = max(0, (resources.memory_bytes - shared) // per_frame)
    worker_limit = min(max(0, usable), int(memory_slots))
    source, requested, status = 'conservative', None, None
    override = os.environ.get('PHASE_PHASE_WORKERS')
    if override is not None:
        requested = int(override)
        if requested < 1:
            raise ValueError('PHASE_PHASE_WORKERS must be positive')
        workers = min(requested, worker_limit)
        source = 'explicit override'
        status = f'requested {requested}; resource-limited to {workers}'
    else:
        workload = (dict(output_workers=resources.output_workers,
                         proposal_workers=resources.proposal_workers if generate_review_proposals else 0,
                         output_pending=resources.pending_outputs, generate_review_proposals=generate_review_proposals)
                    if hasattr(resources, 'output_workers') and hasattr(resources, 'proposal_workers') else None)
        selected, status = calibrated_workers(calibration_path, pixels, worker_limit, workload=workload,
            generate_review_proposals=generate_review_proposals)
        if selected is None:
            workers = min(3, worker_limit)
        else:
            workers, requested, source = selected, selected, 'verified calibration'
    details = dict(selection_source=source, requested_phase_workers=requested,
                   worker_limit=worker_limit,
                   calibration_path=str(calibration_path) if calibration_path is not None else None,
                   calibration_status=status, proposal_estimated_bytes=proposal_bytes,
                   generate_review_proposals=generate_review_proposals)
    if not enabled or workers < 2:
        reason = 'disabled' if not enabled else 'insufficient CPU or memory for overlap'
        if enabled and workers == 1 and source != 'conservative':
            reason = 'selected one phase worker; serial execution retained'
        return ThroughputResources(False, 1, compute_threads, 0, 1,
                                   resources.estimated_working_bytes, reason, **details)
    # A count above three must be explicitly requested or supported by exact
    # measurements for this hardware, image size and unchanged implementation.
    threads = 1
    return ThroughputResources(True, workers, threads, optical, workers,
                               shared + workers * per_frame, 'bounded CUDA/CPU overlap', **details)


def apply_phase_threads(plan):
    """Called once before phase worker threads start, never inside a worker."""
    import cv2
    from threadpoolctl import threadpool_limits
    cv2.setNumThreads(plan.phase_threads)
    threadpool_limits(limits=plan.phase_threads)
