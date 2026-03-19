"""Constellation and satellite data structures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from Basilisk.utilities import macros, orbitalMotion

from aeosbench.constants import MU_EARTH

from .orbits import Orbit

Inertia = tuple[float, float, float, float, float, float, float, float, float]


class SensorType(IntEnum):
    VISIBLE = 1
    NEAR_INFRARED = 2


@dataclass(frozen=True)
class SolarPanel:
    direction: tuple[float, float, float]
    area: float
    efficiency: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SolarPanel":
        direction = payload["direction"]
        if not isinstance(direction, list) or len(direction) != 3:
            raise TypeError("solar panel direction must have length 3")
        return cls(
            direction=(float(direction[0]), float(direction[1]), float(direction[2])),
            area=float(payload["area"]),
            efficiency=float(payload["efficiency"]),
        )

    @property
    def data(self) -> list[float]:
        return [*self.direction, self.area, self.efficiency]


@dataclass(frozen=True)
class Sensor:
    type_: SensorType
    enabled: bool
    half_field_of_view: float
    power: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Sensor":
        return cls(
            type_=SensorType(int(payload["type"])),
            enabled=bool(payload["enabled"]),
            half_field_of_view=float(payload["half_field_of_view"]),
            power=float(payload["power"]),
        )

    @property
    def data(self) -> list[float]:
        return [self.half_field_of_view, self.power]


@dataclass(frozen=True)
class Battery:
    capacity: float
    percentage: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Battery":
        return cls(
            capacity=float(payload["capacity"]),
            percentage=float(payload["percentage"]),
        )

    @property
    def static_data(self) -> list[float]:
        return [self.capacity]

    @property
    def dynamic_data(self) -> list[float]:
        return [self.percentage]


@dataclass(frozen=True)
class ReactionWheel:
    rw_type: str
    rw_direction: tuple[float, float, float]
    max_momentum: float
    rw_speed_init: float
    power: float
    efficiency: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReactionWheel":
        direction = payload["rw_direction"]
        if not isinstance(direction, list) or len(direction) != 3:
            raise TypeError("reaction wheel direction must have length 3")
        return cls(
            rw_type=str(payload["rw_type"]),
            rw_direction=(float(direction[0]), float(direction[1]), float(direction[2])),
            max_momentum=float(payload["max_momentum"]),
            rw_speed_init=float(payload["rw_speed_init"]),
            power=float(payload["power"]),
            efficiency=float(payload["efficiency"]),
        )

    @property
    def static_data(self) -> list[float]:
        return [
            *self.rw_direction,
            self.max_momentum,
            self.power,
            self.efficiency,
        ]

    @property
    def dynamic_data(self) -> list[float]:
        return [self.rw_speed_init]


@dataclass(frozen=True)
class MRPControl:
    k: float
    ki: float
    p: float
    integral_limit: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MRPControl":
        return cls(
            k=float(payload["k"]),
            ki=float(payload["ki"]),
            p=float(payload["p"]),
            integral_limit=float(payload["integral_limit"]),
        )

    @property
    def data(self) -> list[float]:
        return [self.k, self.ki, self.p, self.integral_limit]


@dataclass(frozen=True)
class Satellite:
    id_: int
    inertia: Inertia
    mass: float
    center_of_mass: tuple[float, float, float]
    orbit_id: int
    orbit: Orbit
    solar_panel: SolarPanel
    sensor: Sensor
    battery: Battery
    reaction_wheels: tuple[ReactionWheel, ReactionWheel, ReactionWheel]
    mrp_control: MRPControl
    true_anomaly: float
    mrp_attitude_bn: tuple[float, float, float]

    @classmethod
    def from_dict(cls, payload: dict[str, Any], orbits: dict[int, Orbit]) -> "Satellite":
        inertia = payload["inertia"]
        center_of_mass = payload["center_of_mass"]
        mrp_attitude_bn = payload["mrp_attitude_bn"]
        reaction_wheels = payload["reaction_wheels"]
        if not isinstance(inertia, list) or len(inertia) != 9:
            raise TypeError("satellite inertia must have length 9")
        if not isinstance(center_of_mass, list) or len(center_of_mass) != 3:
            raise TypeError("center_of_mass must have length 3")
        if not isinstance(mrp_attitude_bn, list) or len(mrp_attitude_bn) != 3:
            raise TypeError("mrp_attitude_bn must have length 3")
        if not isinstance(reaction_wheels, list) or len(reaction_wheels) != 3:
            raise TypeError("reaction_wheels must have length 3")
        orbit_id = int(payload["orbit"])
        return cls(
            id_=int(payload["id"]),
            inertia=tuple(float(value) for value in inertia),  # type: ignore[arg-type]
            mass=float(payload["mass"]),
            center_of_mass=(
                float(center_of_mass[0]),
                float(center_of_mass[1]),
                float(center_of_mass[2]),
            ),
            orbit_id=orbit_id,
            orbit=orbits[orbit_id],
            solar_panel=SolarPanel.from_dict(payload["solar_panel"]),
            sensor=Sensor.from_dict(payload["sensor"]),
            battery=Battery.from_dict(payload["battery"]),
            reaction_wheels=tuple(
                ReactionWheel.from_dict(item) for item in reaction_wheels
            ),  # type: ignore[arg-type]
            mrp_control=MRPControl.from_dict(payload["mrp_control"]),
            true_anomaly=float(payload["true_anomaly"]),
            mrp_attitude_bn=(
                float(mrp_attitude_bn[0]),
                float(mrp_attitude_bn[1]),
                float(mrp_attitude_bn[2]),
            ),
        )

    @property
    def rv(self) -> tuple[np.ndarray, np.ndarray]:
        elements = orbitalMotion.ClassicElements()
        elements.e = self.orbit.eccentricity
        elements.a = self.orbit.semi_major_axis
        elements.i = self.orbit.inclination * macros.D2R
        elements.Omega = self.orbit.right_ascension_of_the_ascending_node * macros.D2R
        elements.omega = self.orbit.argument_of_perigee * macros.D2R
        elements.f = self.true_anomaly * macros.D2R
        return orbitalMotion.elem2rv(MU_EARTH, elements)

    @property
    def static_data(self) -> list[float]:
        reaction_wheels: list[float] = []
        for wheel in self.reaction_wheels:
            reaction_wheels.extend(wheel.static_data)
        return [
            *self.inertia,
            self.mass,
            *self.center_of_mass,
            *self.orbit.data,
            *self.solar_panel.data,
            *self.sensor.data,
            *self.battery.static_data,
            *reaction_wheels,
            *self.mrp_control.data,
        ]

    @property
    def dynamic_data(self) -> list[float]:
        reaction_wheels: list[float] = []
        for wheel in self.reaction_wheels:
            reaction_wheels.extend(wheel.dynamic_data)
        return [
            *self.battery.dynamic_data,
            *reaction_wheels,
            self.true_anomaly,
            *self.mrp_attitude_bn,
        ]


class Constellation(dict[int, Satellite]):
    """Mapping of satellite id to current satellite state."""

    @classmethod
    def load(cls, path: str | Path) -> "Constellation":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("constellation payload must be a mapping")
        orbits_payload = payload["orbits"]
        satellites_payload = payload["satellites"]
        if not isinstance(orbits_payload, list) or not isinstance(satellites_payload, list):
            raise TypeError("constellation orbits/satellites must be lists")
        orbits = {
            orbit_payload["id"]: Orbit.from_dict(orbit_payload)
            for orbit_payload in orbits_payload
        }
        return cls(
            {
                int(satellite_payload["id"]): Satellite.from_dict(
                    satellite_payload,
                    orbits,
                )
                for satellite_payload in satellites_payload
            }
        )

    def sort(self) -> list[Satellite]:
        return sorted(self.values(), key=lambda satellite: satellite.id_)

    def static_to_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        satellites = self.sort()
        sensor_type = torch.tensor(
            [int(satellite.sensor.type_) for satellite in satellites],
            dtype=torch.long,
        )
        data = torch.tensor(
            [satellite.static_data for satellite in satellites],
            dtype=torch.float32,
        )
        return sensor_type, data

    def dynamic_to_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        satellites = self.sort()
        sensor_enabled = torch.tensor(
            [int(satellite.sensor.enabled) for satellite in satellites],
            dtype=torch.long,
        )
        data = torch.tensor(
            [satellite.dynamic_data for satellite in satellites],
            dtype=torch.float32,
        )
        return sensor_enabled, data
