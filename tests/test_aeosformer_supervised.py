import torch

from aeosbench.models.aeosformer import AEOSFormerActor


def test_forward_supervised_uses_raw_feasibility_logits(monkeypatch):
    actor = AEOSFormerActor(
        time_embedding_dim=16,
        sensor_type_embedding_dim=16,
        tasks_data_embedding_dim=16,
        encoder_width=64,
        encoder_depth=2,
        encoder_num_heads=4,
        sensor_enabled_embedding_dim=16,
        constellation_data_embedding_dim=16,
        decoder_width=64,
        decoder_depth=2,
        decoder_num_heads=4,
        time_model_hidden_dim=64,
    )

    raw_mask = torch.tensor([[[1.5, -2.0]]], dtype=torch.float32)
    projected_mask = torch.tensor([[[9.0, 9.0]]], dtype=torch.float32)
    null_logits = torch.tensor([[0.25]], dtype=torch.float32)
    logits = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)

    def fake_forward_outputs(*args, **kwargs):
        return (
            torch.tensor([[[3.0, 4.0]]], dtype=torch.float32),
            raw_mask,
            projected_mask,
            null_logits,
            logits,
        )

    monkeypatch.setattr(actor._transformer, "forward_outputs", fake_forward_outputs)

    outputs = actor.forward_supervised(
        torch.tensor([0], dtype=torch.long),
        torch.zeros((1, 1), dtype=torch.long),
        torch.zeros((1, 1), dtype=torch.long),
        torch.zeros((1, 1, 56), dtype=torch.float32),
        torch.ones((1, 1), dtype=torch.bool),
        torch.zeros((1, 2), dtype=torch.long),
        torch.zeros((1, 2, 6), dtype=torch.float32),
        torch.ones((1, 2), dtype=torch.bool),
    )

    assert torch.equal(outputs.feasibility_logits, raw_mask)
    assert torch.equal(
        outputs.assignment_logits,
        torch.tensor([[[0.25, 1.0, 2.0]]], dtype=torch.float32),
    )
