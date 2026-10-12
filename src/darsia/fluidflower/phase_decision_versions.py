"""Select an immutable numerical decision version from its recorded profile."""
import json
from pathlib import Path
from . import phase_decision_runtime as legacy

PROFILE_FILE=legacy.PROFILE_FILE
PROFILE_KEY=legacy.PROFILE_KEY
split_profile=legacy.split_profile


def implementation(spec):
    if spec.get('format')==legacy.VERSION:return legacy
    from . import phase_decision_age_runtime as age
    if spec.get('format')==age.VERSION:return age
    from . import phase_regional_runtime as regional
    if spec.get('format')==regional.VERSION:return regional
    raise ValueError('Unknown unified phase decision version')


def attach_profile(root,inherited):
    path=Path(root)/PROFILE_FILE
    if not path.is_file():return inherited
    return implementation(json.loads(path.read_text(encoding='utf-8'))).attach_profile(root,inherited)


def runtime_identity(inherited_identity,spec):
    return implementation(spec).runtime_identity(inherited_identity,spec)


def DecisionSequence(context,provider,spec,checkpoint):
    return implementation(spec).DecisionSequence(context,provider,spec,checkpoint)
