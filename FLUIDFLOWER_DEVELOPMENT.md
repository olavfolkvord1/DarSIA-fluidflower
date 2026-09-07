# FluidFlower application development snapshot

This fork preserves the complete Qt application and supporting workflow modules
under `src/darsia/fluidflower`, in addition to the original DarSIA source and
attribution. It is not the upstream DarSIA release.

The parent project is <https://github.com/jwboth/ff_ac/tree/codex/fluidflower-app-publication-20260907>.
Clone that branch with `--recurse-submodules` to obtain its compatible launchers,
historical training/calibration scripts and environment configuration. The full
application and phase studio remain separate entry points:

```sh
darsia-fluidflower
darsia-fluidflower-phase-studio
```

For incremental adoption into the newer upstream GUI, prefer the smaller optional
modules at <https://github.com/olavfolkvord1/darsia-extensions>. They can be used
from a terminal without installing this complete application.

No experimental image series, trained model weights, reviewer sessions or local
runtime environments are bundled. The application icon is included as a UI asset.
Software tests do not establish scientific validity of a particular trained model
or experiment. Preserve upstream and extracted-source license notices when reusing
code; publication is not relicensing.
