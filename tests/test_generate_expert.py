import torch

from aeosbench.generation.expert import GreedyExpertScheduler


class _FakeTask:
    def __init__(self, id_: int) -> None:
        self.id_ = id_


class _FakeConstellation:
    def __init__(self, satellites):
        self._satellites = satellites

    def sort(self):
        return self._satellites


class _FakeSatellite:
    def __init__(self, x: float) -> None:
        self.rv = (torch.tensor([x, 0.0, 0.0]), torch.zeros(3))


class _FakeTimer:
    def __init__(self) -> None:
        self.time = 0


class _FakeEnvironment:
    def __init__(self) -> None:
        self.timer = _FakeTimer()
        self._constellation = _FakeConstellation([_FakeSatellite(0.0), _FakeSatellite(10.0)])

    def get_constellation(self):
        return self._constellation


class _FakeTaskManager:
    def __init__(self) -> None:
        self.ongoing_tasks = [_FakeTask(10), _FakeTask(20)]


def test_greedy_expert_scheduler_keeps_sticky_assignment_and_resets_on_idle(monkeypatch):
    environment = _FakeEnvironment()
    task_manager = _FakeTaskManager()
    scheduler = GreedyExpertScheduler()

    def fake_task_positions(taskset, *, time_step):
        del taskset
        if time_step == 0:
            return torch.tensor([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        return torch.tensor([[9.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    monkeypatch.setattr("aeosbench.generation.expert._task_positions_eci", fake_task_positions)
    monkeypatch.setattr(
        "aeosbench.generation.expert._check_constraints",
        lambda distance, orbital_radius: torch.ones_like(distance, dtype=torch.bool),
    )

    assignment = scheduler.step(environment=environment, task_manager=task_manager)
    assert assignment == [0, 1]

    environment.timer.time = 1
    assignment = scheduler.step(environment=environment, task_manager=task_manager)
    assert assignment == [0, 1]

    task_manager.ongoing_tasks = []
    assignment = scheduler.step(environment=environment, task_manager=task_manager)
    assert assignment == [-1, -1]
