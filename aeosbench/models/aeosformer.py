"""Shared AEOS-Former implementation for evaluation and supervised training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from aeosbench.constants import MAX_TIME_STEP, SATELLITE_DIM, TASK_DIM
from aeosbench.data import SensorType

TIME_SCALE = 50.0


@dataclass(frozen=True)
class SupervisedOutputs:
    feasibility_logits: torch.Tensor
    timing_predictions: torch.Tensor
    assignment_logits: torch.Tensor


def sinusoidal_position_embedding(positions: torch.Tensor, width: int) -> torch.Tensor:
    if width % 2 != 0:
        raise ValueError("sinusoidal position embedding requires an even width")
    frequencies = torch.linspace(0, 1, width // 2, device=positions.device)
    scaled = positions.to(dtype=torch.float32).unsqueeze(-1) / (10000 ** frequencies)
    return torch.stack((scaled.sin(), scaled.cos()), dim=-1).reshape(*scaled.shape[:-1], width)


class TransformerBlock(nn.Module):
    """Transformer block with state-dict-compatible attribute names."""

    def __init__(self, *, width: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self._norm1 = nn.LayerNorm(width, eps=1e-6)
        self._attention = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self._norm2 = nn.LayerNorm(width, eps=1e-6)
        self._mlp = nn.Sequential(
            nn.Linear(width, int(width * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(width * mlp_ratio), width),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm = self._norm1(x)
        attention, _ = self._attention(
            norm,
            norm,
            norm,
            need_weights=False,
            attn_mask=attention_mask,
        )
        x = x + attention
        x = x + self._mlp(self._norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        time_embedding_dim: int,
        sensor_type_embedding_dim: int,
        data_embedding_dim: int,
        width: int,
        depth: int,
        num_heads: int,
        data_dim: int = TASK_DIM,
    ) -> None:
        super().__init__()
        self._num_heads = num_heads
        self._data_embedding = nn.Linear(data_dim, data_embedding_dim)
        self._in_projector = nn.Linear(
            time_embedding_dim + sensor_type_embedding_dim + data_embedding_dim,
            width,
        )
        self._blocks = nn.ModuleList(
            [TransformerBlock(width=width, num_heads=num_heads) for _ in range(depth)]
        )
        self._norm = nn.LayerNorm(width)

    def forward(
        self,
        time_embedding: torch.Tensor,
        sensor_type_embedding: torch.Tensor,
        data: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        repeated_time = time_embedding.unsqueeze(1).expand(-1, data.shape[1], -1)
        data_embedding = self._data_embedding(data)
        x = self._in_projector(
            torch.cat((repeated_time, sensor_type_embedding, data_embedding), dim=-1)
        )
        pair_mask = attention_mask.unsqueeze(-1) & attention_mask.unsqueeze(-2)
        expanded_mask = pair_mask.unsqueeze(1).expand(-1, self._num_heads, -1, -1)
        expanded_mask = expanded_mask.reshape(-1, attention_mask.shape[1], attention_mask.shape[1])
        attn_mask = torch.where(
            expanded_mask,
            torch.zeros((), device=x.device, dtype=x.dtype),
            torch.full((), float("-inf"), device=x.device, dtype=x.dtype),
        )
        for block in self._blocks:
            x = block(x, attention_mask=attn_mask)
        return self._norm(x)


class DecoderBlock(TransformerBlock):
    def __init__(self, *, width: int, num_heads: int) -> None:
        super().__init__(width=width, num_heads=num_heads)
        self._norm3 = nn.LayerNorm(width, eps=1e-6)
        self._cross_attention = nn.MultiheadAttention(width, num_heads, batch_first=True)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        hidden_states: torch.Tensor,
        cross_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = super().forward(x, attention_mask=attention_mask)
        norm = self._norm3(x)
        cross_attention, _ = self._cross_attention(
            norm,
            hidden_states,
            hidden_states,
            need_weights=False,
            attn_mask=cross_attention_mask,
        )
        return x + cross_attention


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        time_embedding_dim: int,
        sensor_type_embedding_dim: int,
        sensor_enabled_embedding_dim: int,
        data_embedding_dim: int,
        width: int,
        depth: int,
        num_heads: int,
        data_dim: int = SATELLITE_DIM,
    ) -> None:
        super().__init__()
        self._num_heads = num_heads
        self._sensor_enabled_embedding = nn.Embedding(2, sensor_enabled_embedding_dim)
        self._data_embedding = nn.Linear(data_dim, data_embedding_dim)
        self._in_projector = nn.Linear(
            time_embedding_dim
            + sensor_type_embedding_dim
            + sensor_enabled_embedding_dim
            + data_embedding_dim,
            width,
        )
        self._blocks = nn.ModuleList(
            [DecoderBlock(width=width, num_heads=num_heads) for _ in range(depth)]
        )
        self._norm = nn.LayerNorm(width)
        self._null_task = nn.Parameter(torch.zeros(width))

    def forward(
        self,
        time_embedding: torch.Tensor,
        sensor_type_embedding: torch.Tensor,
        sensor_enabled: torch.Tensor,
        data: torch.Tensor,
        mask: torch.Tensor,
        hidden_states: torch.Tensor,
        tasks_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        repeated_time = time_embedding.unsqueeze(1).expand(-1, data.shape[1], -1)
        enabled_embedding = self._sensor_enabled_embedding(sensor_enabled)
        data_embedding = self._data_embedding(data)
        x = self._in_projector(
            torch.cat(
                (repeated_time, sensor_type_embedding, enabled_embedding, data_embedding),
                dim=-1,
            )
        )

        self_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        expanded_self_mask = self_mask.unsqueeze(1).expand(-1, self._num_heads, -1, -1)
        expanded_self_mask = expanded_self_mask.reshape(-1, mask.shape[1], mask.shape[1])
        attention_mask = torch.where(
            expanded_self_mask,
            torch.zeros((), device=x.device, dtype=x.dtype),
            torch.full((), float("-inf"), device=x.device, dtype=x.dtype),
        )

        cross_mask = tasks_mask.unsqueeze(1).expand(-1, data.shape[1], -1)
        cross_mask = torch.where(
            cross_mask,
            time_mask,
            torch.full((), float("-inf"), device=x.device, dtype=x.dtype),
        )
        cross_mask = cross_mask.unsqueeze(1).expand(-1, self._num_heads, -1, -1)
        cross_mask = cross_mask.reshape(-1, data.shape[1], hidden_states.shape[1])

        for block in self._blocks:
            x = block(
                x,
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                cross_attention_mask=cross_mask,
            )
        x = self._norm(x)

        null_logits = torch.einsum("bsd,d->bs", x, self._null_task)
        logits = torch.einsum("bsd,btd->bst", x, hidden_states)
        logits_mask = torch.where(
            tasks_mask.unsqueeze(1),
            torch.zeros((), device=x.device, dtype=x.dtype),
            torch.full((), float("-inf"), device=x.device, dtype=x.dtype),
        )
        logits = logits + logits_mask
        return null_logits, logits


class TimeModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = SATELLITE_DIM + TASK_DIM,
        time_embedding_dim: int = 64,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        time_embedding = sinusoidal_position_embedding(
            torch.arange(MAX_TIME_STEP),
            time_embedding_dim,
        )
        self._time_embedding = nn.Parameter(time_embedding)
        self._mlp = nn.Sequential(
            nn.Linear(input_dim + time_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def _predict(
        self,
        time_steps: torch.Tensor,
        constellation_data: torch.Tensor,
        tasks_data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_embedding = self._time_embedding[time_steps]
        x = self._mlp(torch.cat((constellation_data, tasks_data, time_embedding), dim=-1))
        return x.unbind(-1)

    def predict(
        self,
        time_steps: torch.Tensor | Iterable[int],
        constellation_data: torch.Tensor,
        constellation_mask: torch.Tensor,
        tasks_data: torch.Tensor,
        tasks_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ns = constellation_data.shape[1]
        nt = tasks_data.shape[1]
        mask = constellation_mask.unsqueeze(-1) & tasks_mask.unsqueeze(-2)

        if not isinstance(time_steps, torch.Tensor):
            time_steps = torch.tensor(list(time_steps), dtype=torch.long, device=mask.device)
        if time_steps.ndim == 0:
            time_steps = time_steps.unsqueeze(0)

        batch_indices, satellite_indices, task_indices = mask.nonzero(as_tuple=True)
        flat_time_steps = time_steps[batch_indices]
        flat_constellation = constellation_data[batch_indices, satellite_indices]
        flat_tasks = tasks_data[batch_indices, task_indices]

        pred_time, pred_mask = self._predict(flat_time_steps, flat_constellation, flat_tasks)
        padded_pred_time = torch.full(
            (mask.shape[0], ns, nt),
            -1.0,
            dtype=pred_time.dtype,
            device=pred_time.device,
        )
        padded_pred_mask = torch.full(
            (mask.shape[0], ns, nt),
            float("-inf"),
            dtype=pred_mask.dtype,
            device=pred_mask.device,
        )
        padded_pred_time[mask] = pred_time
        padded_pred_mask[mask] = pred_mask
        return padded_pred_time * TIME_SCALE, padded_pred_mask


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        time_embedding_dim: int = 64,
        sensor_type_embedding_dim: int,
        tasks_data_embedding_dim: int,
        encoder_width: int,
        encoder_depth: int,
        encoder_num_heads: int,
        sensor_enabled_embedding_dim: int,
        constellation_data_embedding_dim: int,
        decoder_width: int,
        decoder_depth: int,
        decoder_num_heads: int,
        time_model_hidden_dim: int,
    ) -> None:
        super().__init__()
        time_embedding = sinusoidal_position_embedding(
            torch.arange(MAX_TIME_STEP),
            time_embedding_dim,
        )
        self._time_embedding = nn.Parameter(time_embedding)
        self._sensor_type_embedding = nn.Embedding(len(SensorType), sensor_type_embedding_dim)
        self._encoder = Encoder(
            time_embedding_dim=time_embedding_dim,
            sensor_type_embedding_dim=sensor_type_embedding_dim,
            data_embedding_dim=tasks_data_embedding_dim,
            width=encoder_width,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
        )
        self._decoder = Decoder(
            time_embedding_dim=time_embedding_dim,
            sensor_type_embedding_dim=sensor_type_embedding_dim,
            sensor_enabled_embedding_dim=sensor_enabled_embedding_dim,
            data_embedding_dim=constellation_data_embedding_dim,
            width=decoder_width,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
        )
        self._time_model = TimeModel(
            time_embedding_dim=time_embedding_dim,
            hidden_dim=time_model_hidden_dim,
        )
        self._time_projection = nn.Linear(1, 1)

    def forward_outputs(
        self,
        time_steps: torch.Tensor | Iterable[int],
        constellation_sensor_type: torch.Tensor,
        constellation_sensor_enabled: torch.Tensor,
        constellation_data: torch.Tensor,
        constellation_mask: torch.Tensor,
        tasks_sensor_type: torch.Tensor,
        tasks_data: torch.Tensor,
        tasks_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(time_steps, torch.Tensor):
            flat_time_steps = time_steps.flatten()
        else:
            flat_time_steps = torch.tensor(list(time_steps), dtype=torch.long, device=tasks_data.device)

        pred_time, raw_feasibility_logits = self._time_model.predict(
            flat_time_steps,
            constellation_data,
            constellation_mask,
            tasks_data,
            tasks_mask,
        )
        time_mask = raw_feasibility_logits.clamp_min(-100.0)
        time_mask = self._time_projection(time_mask.unsqueeze(-1)).squeeze(-1)

        time_embedding = self._time_embedding[flat_time_steps]
        task_sensor_embedding = self._sensor_type_embedding(tasks_sensor_type)
        hidden_states = self._encoder(
            time_embedding,
            task_sensor_embedding,
            tasks_data,
            tasks_mask,
        )
        constellation_sensor_embedding = self._sensor_type_embedding(
            constellation_sensor_type
        )
        null_logits, logits = self._decoder(
            time_embedding,
            constellation_sensor_embedding,
            constellation_sensor_enabled,
            constellation_data,
            constellation_mask,
            hidden_states,
            tasks_mask,
            time_mask,
        )
        return pred_time, raw_feasibility_logits, time_mask, null_logits, logits

    def forward(
        self,
        time_steps: torch.Tensor | Iterable[int],
        constellation_sensor_type: torch.Tensor,
        constellation_sensor_enabled: torch.Tensor,
        constellation_data: torch.Tensor,
        constellation_mask: torch.Tensor,
        tasks_sensor_type: torch.Tensor,
        tasks_data: torch.Tensor,
        tasks_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, _, null_logits, logits = self.forward_outputs(
            time_steps,
            constellation_sensor_type,
            constellation_sensor_enabled,
            constellation_data,
            constellation_mask,
            tasks_sensor_type,
            tasks_data,
            tasks_mask,
        )
        return null_logits, logits


class AEOSFormerActor(nn.Module):
    def __init__(
        self,
        *,
        time_embedding_dim: int = 64,
        sensor_type_embedding_dim: int = 128,
        tasks_data_embedding_dim: int = 128,
        encoder_width: int = 512,
        encoder_depth: int = 12,
        encoder_num_heads: int = 16,
        sensor_enabled_embedding_dim: int = 128,
        constellation_data_embedding_dim: int = 128,
        decoder_width: int = 512,
        decoder_depth: int = 12,
        decoder_num_heads: int = 16,
        time_model_hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self._transformer = Transformer(
            time_embedding_dim=time_embedding_dim,
            sensor_type_embedding_dim=sensor_type_embedding_dim,
            tasks_data_embedding_dim=tasks_data_embedding_dim,
            encoder_width=encoder_width,
            encoder_depth=encoder_depth,
            encoder_num_heads=encoder_num_heads,
            sensor_enabled_embedding_dim=sensor_enabled_embedding_dim,
            constellation_data_embedding_dim=constellation_data_embedding_dim,
            decoder_width=decoder_width,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            time_model_hidden_dim=time_model_hidden_dim,
        )

    def forward_supervised(
        self,
        time_steps: torch.Tensor | Iterable[int],
        constellation_sensor_type: torch.Tensor,
        constellation_sensor_enabled: torch.Tensor,
        constellation_data: torch.Tensor,
        constellation_mask: torch.Tensor,
        tasks_sensor_type: torch.Tensor,
        tasks_data: torch.Tensor,
        tasks_mask: torch.Tensor,
    ) -> SupervisedOutputs:
        pred_time, feasibility_logits, _, null_logits, logits = self._transformer.forward_outputs(
            time_steps,
            constellation_sensor_type,
            constellation_sensor_enabled,
            constellation_data,
            constellation_mask,
            tasks_sensor_type,
            tasks_data,
            tasks_mask,
        )
        return SupervisedOutputs(
            feasibility_logits=feasibility_logits,
            timing_predictions=pred_time,
            assignment_logits=torch.cat((null_logits.unsqueeze(-1), logits), dim=-1),
        )

    @torch.no_grad()
    def predict(
        self,
        time_steps: torch.Tensor | Iterable[int],
        constellation_sensor_type: torch.Tensor,
        constellation_sensor_enabled: torch.Tensor,
        constellation_data: torch.Tensor,
        constellation_mask: torch.Tensor,
        tasks_sensor_type: torch.Tensor,
        tasks_data: torch.Tensor,
        tasks_mask: torch.Tensor,
    ) -> torch.Tensor:
        null_logits, logits = self._transformer(
            time_steps,
            constellation_sensor_type,
            constellation_sensor_enabled,
            constellation_data,
            constellation_mask,
            tasks_sensor_type,
            tasks_data,
            tasks_mask,
        )
        return torch.cat((null_logits.unsqueeze(-1), logits), dim=-1)
