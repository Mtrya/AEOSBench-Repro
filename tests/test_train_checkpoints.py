import torch

from aeosbench.evaluation.checkpoints import build_actor, load_actor_checkpoint
from aeosbench.training.checkpoints import load_checkpoint, save_checkpoint
from aeosbench.training.config import load_training_config


def test_training_checkpoint_roundtrip_preserves_evaluator_compatibility(tmp_path):
    config = load_training_config("configs/train_sl/tiny.yaml")
    model = build_actor(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, total_iters=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    checkpoint_dir = save_checkpoint(
        work_dir=tmp_path,
        iteration=1,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        scaler=scaler,
        seed=3407,
        autocast=False,
        config_path=config.path,
        config_hash=config.hash_,
    )

    meta = load_checkpoint(
        checkpoint_dir,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        scaler=scaler,
    )
    reloaded = load_actor_checkpoint(config.model, checkpoint_dir / "model.pth")

    assert int(meta["iter"]) == 1
    assert checkpoint_dir.joinpath("config.yaml").exists()
    assert checkpoint_dir.joinpath("scaler.pth").exists()
    assert isinstance(reloaded, torch.nn.Module)
