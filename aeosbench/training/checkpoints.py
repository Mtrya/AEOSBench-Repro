"""Checkpoint helpers for supervised training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def checkpoint_dir(work_dir: Path, iteration: int) -> Path:
    return work_dir / "checkpoints" / f"iter_{iteration}"


def resolve_resume_path(work_dir: Path, resume: str | Path | None) -> Path | None:
    if resume is None:
        return None
    if str(resume) != "latest":
        return Path(resume)
    checkpoints_dir = work_dir / "checkpoints"
    candidates = sorted(checkpoints_dir.glob("iter_*"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: int(path.name.split("_", 1)[1]))


def save_checkpoint(
    *,
    work_dir: Path,
    iteration: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    seed: int,
    autocast: bool,
    config_path: Path,
    config_hash: str,
) -> Path:
    target_dir = checkpoint_dir(work_dir, iteration)
    target_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target_dir / "model.pth")
    torch.save(optimizer.state_dict(), target_dir / "optimizer.pth")
    torch.save(lr_scheduler.state_dict(), target_dir / "lr_scheduler.pth")
    torch.save(scaler.state_dict(), target_dir / "scaler.pth")
    torch.save(
        {
            "iter": iteration,
            "seed": seed,
            "autocast": autocast,
            "config_hash": config_hash,
            "config_path": str(config_path),
        },
        target_dir / "meta.pth",
    )
    (target_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return target_dir


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
    checkpoint_dir_path = Path(checkpoint_path)
    if not checkpoint_dir_path.exists():
        raise FileNotFoundError(checkpoint_dir_path)
    if not checkpoint_dir_path.is_dir():
        raise ValueError(f"Checkpoint path must be a directory: {checkpoint_dir_path}")

    model.load_state_dict(
        torch.load(checkpoint_dir_path / "model.pth", map_location="cpu", weights_only=False),
    )
    optimizer.load_state_dict(
        torch.load(checkpoint_dir_path / "optimizer.pth", map_location="cpu", weights_only=False),
    )
    lr_scheduler.load_state_dict(
        torch.load(checkpoint_dir_path / "lr_scheduler.pth", map_location="cpu", weights_only=False),
    )
    scaler.load_state_dict(
        torch.load(checkpoint_dir_path / "scaler.pth", map_location="cpu", weights_only=False),
    )
    meta = torch.load(checkpoint_dir_path / "meta.pth", map_location="cpu", weights_only=False)
    if not isinstance(meta, dict):
        raise TypeError(f"unsupported checkpoint meta payload: {type(meta)!r}")
    return meta
