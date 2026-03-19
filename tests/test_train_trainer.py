import torch

from aeosbench.training.trainer import _step_optimizer_and_scheduler


class _FakeScaler:
    def __init__(self, scales: list[float]) -> None:
        self._scales = scales
        self._index = 0
        self.step_calls = 0
        self.update_calls = 0

    def get_scale(self) -> float:
        return self._scales[self._index]

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.step_calls += 1

    def update(self) -> None:
        self.update_calls += 1
        if self._index + 1 < len(self._scales):
            self._index += 1


class _FakeScheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1


def test_step_optimizer_and_scheduler_advances_scheduler_without_amp():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-4)
    scheduler = _FakeScheduler()
    scaler = _FakeScaler([1.0, 1.0])

    stepped = _step_optimizer_and_scheduler(
        optimizer=optimizer,
        lr_scheduler=scheduler,  # type: ignore[arg-type]
        scaler=scaler,  # type: ignore[arg-type]
        autocast_enabled=False,
    )

    assert stepped is True
    assert scheduler.step_calls == 1
    assert scaler.step_calls == 0
    assert scaler.update_calls == 0


def test_step_optimizer_and_scheduler_skips_scheduler_when_amp_step_is_skipped():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-4)
    scheduler = _FakeScheduler()
    scaler = _FakeScaler([65536.0, 32768.0])

    stepped = _step_optimizer_and_scheduler(
        optimizer=optimizer,
        lr_scheduler=scheduler,  # type: ignore[arg-type]
        scaler=scaler,  # type: ignore[arg-type]
        autocast_enabled=True,
    )

    assert stepped is False
    assert scheduler.step_calls == 0
    assert scaler.step_calls == 1
    assert scaler.update_calls == 1


def test_step_optimizer_and_scheduler_advances_scheduler_when_amp_step_succeeds():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-4)
    scheduler = _FakeScheduler()
    scaler = _FakeScaler([32768.0, 32768.0])

    stepped = _step_optimizer_and_scheduler(
        optimizer=optimizer,
        lr_scheduler=scheduler,  # type: ignore[arg-type]
        scaler=scaler,  # type: ignore[arg-type]
        autocast_enabled=True,
    )

    assert stepped is True
    assert scheduler.step_calls == 1
    assert scaler.step_calls == 1
    assert scaler.update_calls == 1
