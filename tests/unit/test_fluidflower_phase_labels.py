from darsia.fluidflower.phase_labels import (
    VISIBLE_PHASE_LABEL_SCHEMA,
    assess_model_phase,
    phase_choice_from_record,
    visible_phase_payload,
)


def test_dominant_labels_retain_both_visible_co2_phases() -> None:
    aqueous = visible_phase_payload("predominantly_aqueous_co2")
    gas = visible_phase_payload("predominantly_free_gas")

    assert aqueous["phase_label_schema"] == VISIBLE_PHASE_LABEL_SCHEMA
    assert aqueous["phase_class"] == "mixed_gas_aqueous"
    assert aqueous["phase_dominance"] == "aqueous"
    assert aqueous["phase_components"] == ["aqueous", "gas"]
    assert gas["phase_class"] == "mixed_gas_aqueous"
    assert gas["phase_dominance"] == "gas"


def test_pure_free_gas_is_distinct_from_gas_dominant_mixture() -> None:
    pure = visible_phase_payload("free_gas_only")

    assert pure["phase_class"] == "gas"
    assert pure["phase_components"] == ["gas"]


def test_model_disagreement_is_derived_from_human_phase_components() -> None:
    assessment = assess_model_phase("gas", "only_aqueous_co2")

    assert assessment["model_assessment"] == "phase_substitution"
    assert assessment["missing_phase_components"] == ["aqueous"]
    assert assessment["unsupported_phase_components"] == ["gas"]


def test_manual_and_blind_records_resolve_to_same_choice() -> None:
    assert phase_choice_from_record(
        {"category": "confirmed_predominantly_free_gas"}
    ) == "predominantly_free_gas"
    assert phase_choice_from_record(
        {"human_choice": "predominantly_free_gas"}
    ) == "predominantly_free_gas"

