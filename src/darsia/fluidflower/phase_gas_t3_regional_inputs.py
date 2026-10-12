"""Explicit G3-T3 opt-in to the unchanged source-exact gas feature pipeline.

No global variables are patched. T2 callers retain their original validation
and metadata defaults; T3 opts into its own converged four-feature pixel head.
"""
from . import phase_gas_t2_regional_inputs as shared

CONTRACT = "ffac-g3-t3-source-exact-regional-inputs-v1"
SPACING_M = shared.SPACING_M
FLATFIELD_FEATURE_NAMES = shared.FLATFIELD_FEATURE_NAMES
prepare_frame_inputs = shared.prepare_frame_inputs
replace_gas_from_scores = shared.replace_gas_from_scores


def verify_pixel_model(path, expected_sha256):
    return shared.verify_pixel_model(path, expected_sha256, training_revision="G3-T3")


def prepare_gas_inputs(*args, **kwargs):
    return shared.prepare_gas_inputs(*args, **kwargs, input_contract=CONTRACT)


def evaluate_gas_frame(*args, **kwargs):
    return shared.evaluate_gas_frame(*args, **kwargs, training_revision="G3-T3", input_contract=CONTRACT)


def source_evaluator(binding, source, threads):
    return shared.source_evaluator(binding, source, threads,
                                   training_revision="G3-T3", input_contract=CONTRACT)
