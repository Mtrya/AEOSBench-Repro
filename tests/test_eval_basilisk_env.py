from aeosbench.data import SensorType, Task, TaskSet
from aeosbench.evaluation.basilisk_env import BasiliskEnvironment


class _FakeSatellite:
    def __init__(self, satellite_id: int, enabled: bool) -> None:
        self.id_ = satellite_id
        self._enabled = enabled
        self.toggled = 0
        self.guided_targets: list[tuple[float, float] | None] = []

    def to_satellite(self):
        sensor = type("SensorState", (), {"enabled": self._enabled})()
        return type("SatelliteState", (), {"sensor": sensor})()

    def toggle(self) -> None:
        self._enabled = not self._enabled
        self.toggled += 1

    def guide_attitude(self, target_location: tuple[float, float] | None) -> None:
        self.guided_targets.append(target_location)


def test_apply_assignment_uses_satellite_id_order() -> None:
    environment = BasiliskEnvironment.__new__(BasiliskEnvironment)
    low_id = _FakeSatellite(1, enabled=False)
    high_id = _FakeSatellite(5, enabled=False)
    environment._satellites = [high_id, low_id]

    ongoing_tasks = TaskSet(
        [
            Task(
                id_=10,
                release_time=0,
                due_time=10,
                duration=1,
                coordinate=(1.0, 2.0),
                sensor_type=SensorType.VISIBLE,
            ),
            Task(
                id_=11,
                release_time=0,
                due_time=10,
                duration=1,
                coordinate=(3.0, 4.0),
                sensor_type=SensorType.VISIBLE,
            ),
        ]
    )

    environment.apply_assignment([0, 1], ongoing_tasks)

    assert low_id.guided_targets == [(1.0, 2.0)]
    assert high_id.guided_targets == [(3.0, 4.0)]
    assert low_id.toggled == 1
    assert high_id.toggled == 1
