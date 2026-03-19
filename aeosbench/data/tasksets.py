"""Task and task-set data structures."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch

from aeosbench.constants import ECCENTRICITY_EARTH, RADIUS_EARTH

from .constellations import SensorType

Coordinate = tuple[float, float]


@dataclass(frozen=True)
class Task:
    """A single earth observation task."""

    id_: int
    release_time: int
    due_time: int
    duration: int
    coordinate: Coordinate
    sensor_type: SensorType

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Task":
        coordinate = payload["coordinate"]
        if not isinstance(coordinate, list) or len(coordinate) != 2:
            raise TypeError("task coordinate must be a [lat, lon] list")
        return cls(
            id_=int(payload["id"]),
            release_time=int(payload["release_time"]),
            due_time=int(payload["due_time"]),
            duration=int(payload["duration"]),
            coordinate=(float(coordinate[0]), float(coordinate[1])),
            sensor_type=SensorType(int(payload["sensor_type"])),
        )

    @property
    def data(self) -> list[float]:
        latitude, longitude = self.coordinate
        return [
            float(self.release_time),
            float(self.due_time),
            float(self.duration),
            latitude,
            longitude,
        ]

    @property
    def coordinate_ecef(self) -> tuple[float, float, float]:
        latitude = math.radians(self.coordinate[0])
        longitude = math.radians(self.coordinate[1])
        altitude = 0.0
        n_value = RADIUS_EARTH / math.sqrt(
            1.0 - ECCENTRICITY_EARTH * math.sin(latitude) ** 2,
        )
        return (
            (n_value + altitude) * math.cos(latitude) * math.cos(longitude),
            (n_value + altitude) * math.cos(latitude) * math.sin(longitude),
            ((1.0 - ECCENTRICITY_EARTH) * n_value + altitude) * math.sin(latitude),
        )


class TaskSet(list[Task]):
    """Ordered list of tasks with tensor helpers."""

    @classmethod
    def load(cls, path: str | Path) -> "TaskSet":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError("taskset payload must be a list")
        return cls(Task.from_dict(item) for item in payload)

    def __init__(self, tasks: Iterable[Task] = ()) -> None:
        super().__init__(tasks)

    @property
    def release_times(self) -> torch.Tensor:
        return torch.tensor([task.release_time for task in self], dtype=torch.float32)

    @property
    def durations(self) -> torch.Tensor:
        return torch.tensor([task.duration for task in self], dtype=torch.float32)

    def to_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        sensor_type = torch.tensor(
            [int(task.sensor_type) for task in self],
            dtype=torch.long,
        )
        data = torch.tensor([task.data for task in self], dtype=torch.float32)
        return sensor_type, data
