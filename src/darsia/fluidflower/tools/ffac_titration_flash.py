"""ff_ac-owned titration flash model.

Keeping this extension in ff_ac avoids requiring write access to the DarSIA
submodule and makes distributed workers use the same implementation after a
normal ff_ac pull.
"""

from __future__ import annotations

from typing import Any

import darsia
import numpy as np


def alkalinity_from_initial_pH(
    initial_pH: float,
    btb_M: float,
    pKaB: float = 7.10,
) -> float:
    """Return the effective alkalinity reproducing a zero-DIC initial pH.

    The titration model's charge balance contains water and BTB explicitly.
    This helper converts an experimentally interpretable initial pH into the
    equivalent alkalinity used by that balance. It is intentionally opt-in so
    existing serialized calibrations retain their original numerical contract.
    """

    resolved_pH = float(initial_pH)
    resolved_btb = float(btb_M)
    resolved_pka = float(pKaB)
    if not np.isfinite(resolved_pH) or not 0.5 < resolved_pH < 13.5:
        raise ValueError("initial_pH must be finite and between 0.5 and 13.5")
    if not np.isfinite(resolved_btb) or resolved_btb < 0.0:
        raise ValueError("btb_M must be finite and non-negative")
    hydrogen = 10.0**-resolved_pH
    hydroxide = 1.0e-14 / hydrogen
    kab = 10.0**-resolved_pka
    deprotonated_btb = resolved_btb * kab / (kab + hydrogen)
    return float(hydroxide + deprotonated_btb - hydrogen)


def _titration_yellow_fraction(
    dic_m: np.ndarray,
    alkalinity_m: float,
    btb_m: float,
    pka1: float,
    pka2: float,
    pkab: float,
) -> np.ndarray:
    """Return the protonated BTB fraction for a dissolved-carbon grid."""

    kw = 1e-14
    ka1 = 10.0**-pka1
    ka2 = 10.0**-pka2
    kab = 10.0**-pkab
    concentrations = np.asarray(dic_m, dtype=np.float64)
    result = np.empty_like(concentrations)
    for index, total_carbon in enumerate(concentrations):
        low, high = 0.5, 13.5
        for _ in range(60):
            midpoint = 0.5 * (low + high)
            hydrogen = 10.0**-midpoint
            hydroxide = kw / hydrogen
            denominator = (
                hydrogen * hydrogen + ka1 * hydrogen + ka1 * ka2
            )
            balance = (
                alkalinity_m
                + hydrogen
                - hydroxide
                - total_carbon * ka1 * hydrogen / denominator
                - 2.0 * total_carbon * ka1 * ka2 / denominator
                - btb_m * kab / (kab + hydrogen)
            )
            if balance > 0.0:
                low = midpoint
            else:
                high = midpoint
        hydrogen = 10.0 ** (-0.5 * (low + high))
        result[index] = hydrogen / (hydrogen + kab)
    return result


class TitrationFlash(darsia.SimpleFlash):
    """Use a fixed BTB/carbonate titration curve for the aqueous branch."""

    def __init__(
        self,
        min_value_aq: float,
        max_value_aq: float,
        min_value_g: float,
        max_value_g: float,
        restoration: Any = None,
        alkalinity_M: float = 1.25e-3,
        btb_M: float = 0.726e-3,
        pKa1: float = 6.35,
        pKa2: float = 10.33,
        pKaB: float = 7.10,
        co2_sat_M: float = 0.034,
        n_lut: int = 512,
        aqueous_blend: float = 0.0,
        initial_pH: float | None = None,
    ) -> None:
        if initial_pH is not None:
            alkalinity_M = alkalinity_from_initial_pH(
                initial_pH,
                btb_M,
                pKaB,
            )
        super().__init__(
            min_value_aq,
            max_value_aq,
            min_value_g,
            max_value_g,
            restoration,
        )
        self.n_lut = int(n_lut)
        self.aqueous_blend = float(aqueous_blend)
        self.initial_pH = None if initial_pH is None else float(initial_pH)
        if not 0.0 <= self.aqueous_blend <= 1.0:
            raise ValueError("aqueous_blend must be between 0 and 1")
        self.titration_params = {
            "alkalinity_M": alkalinity_M,
            "btb_M": btb_M,
            "pKa1": pKa1,
            "pKa2": pKa2,
            "pKaB": pKaB,
            "co2_sat_M": co2_sat_M,
        }
        self._build_lut(self.n_lut)

    def _build_lut(self, n_lut: int) -> None:
        params = self.titration_params
        saturation = params["co2_sat_M"]
        half = max(8, n_lut // 2)
        total_carbon = np.concatenate(
            [
                np.linspace(0.0, 0.1 * saturation, half, endpoint=False),
                np.linspace(0.1 * saturation, saturation, n_lut - half),
            ]
        )
        yellow = _titration_yellow_fraction(
            total_carbon,
            params["alkalinity_M"],
            params["btb_M"],
            params["pKa1"],
            params["pKa2"],
            params["pKaB"],
        )
        yellow = (yellow - yellow[0]) / (yellow[-1] - yellow[0])
        concentration_aq = total_carbon / saturation
        order = np.argsort(yellow)
        yellow = yellow[order]
        concentration_aq = concentration_aq[order]
        keep = np.concatenate([[True], np.diff(yellow) > 0])
        self._lut_y = yellow[keep]
        self._lut_caq = concentration_aq[keep]

    @darsia.timing_decorator
    def __call__(
        self, signal: darsia.Image
    ) -> tuple[darsia.Image, darsia.Image]:
        denominator = (self.max_value_aq - self.min_value_aq) or 1.0
        yellow = np.clip(
            (signal.img - self.min_value_aq) / denominator,
            0.0,
            1.0,
        )
        physical_concentration = np.interp(
            yellow,
            self._lut_y,
            self._lut_caq,
        )
        concentration_aq = darsia.full_like(
            signal,
            (1.0 - self.aqueous_blend) * physical_concentration
            + self.aqueous_blend * yellow,
        )
        saturation_g = darsia.full_like(
            signal,
            (
                np.clip(signal.img, self.min_value_g, self.max_value_g)
                - self.min_value_g
            )
            / (self.max_value_g - self.min_value_g),
        )
        if self.restoration is not None:
            concentration_aq = self.restoration(concentration_aq)
            saturation_g = self.restoration(saturation_g)
        return concentration_aq, saturation_g

    @classmethod
    def from_simple(
        cls,
        flash: darsia.SimpleFlash,
        **titration_kwargs: Any,
    ) -> "TitrationFlash":
        return cls(
            min_value_aq=flash.min_value_aq,
            max_value_aq=flash.max_value_aq,
            min_value_g=flash.min_value_g,
            max_value_g=flash.max_value_g,
            restoration=getattr(flash, "restoration", None),
            **titration_kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "model": "titration",
                "titration_params": dict(self.titration_params),
                "n_lut": self.n_lut,
                "aqueous_blend": self.aqueous_blend,
                "initial_pH": self.initial_pH,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TitrationFlash":
        return cls(
            min_value_aq=data["min_value_aq"],
            max_value_aq=data["max_value_aq"],
            min_value_g=data["min_value_g"],
            max_value_g=data["max_value_g"],
            n_lut=int(data.get("n_lut", 512)),
            aqueous_blend=float(data.get("aqueous_blend", 0.0)),
            initial_pH=data.get("initial_pH"),
            **dict(data.get("titration_params") or {}),
        )
