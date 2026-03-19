"""Orbit data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Orbit:
    """Orbital elements for a single satellite orbit."""

    id_: int
    eccentricity: float
    semi_major_axis: float
    inclination: float
    right_ascension_of_the_ascending_node: float
    argument_of_perigee: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Orbit":
        return cls(
            id_=int(payload["id"]),
            eccentricity=float(payload["eccentricity"]),
            semi_major_axis=float(payload["semi_major_axis"]),
            inclination=float(payload["inclination"]),
            right_ascension_of_the_ascending_node=float(
                payload["right_ascension_of_the_ascending_node"]
            ),
            argument_of_perigee=float(payload["argument_of_perigee"]),
        )

    @property
    def data(self) -> list[float]:
        return [
            self.eccentricity,
            self.semi_major_axis,
            self.inclination,
            self.right_ascension_of_the_ascending_node,
            self.argument_of_perigee,
        ]
