"""Stable-Baselines3 policy definitions for AEOSBench PPO."""

from __future__ import annotations

from typing import NamedTuple, TypedDict, cast

from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
from torch.distributions import Categorical

from aeosbench.constants import SATELLITE_DIM, TASK_DIM
from aeosbench.evaluation.checkpoints import build_actor, normalized_state_dict
from aeosbench.evaluation.model_config import AEOSFormerConfig
from aeosbench.training.rl_env import MAX_NUM_SATELLITES, MAX_NUM_TASKS

VALUE_WIDTH = 64


class Observation(TypedDict):
    num_satellites: torch.Tensor
    num_tasks: torch.Tensor
    time_step: torch.Tensor
    constellation_sensor_type: torch.Tensor
    constellation_sensor_enabled: torch.Tensor
    constellation_data: torch.Tensor
    tasks_sensor_type: torch.Tensor
    tasks_data: torch.Tensor


class Batch(NamedTuple):
    time_step: torch.Tensor
    constellation_sensor_type: torch.Tensor
    constellation_sensor_enabled: torch.Tensor
    constellation_data: torch.Tensor
    constellation_mask: torch.Tensor
    tasks_sensor_type: torch.Tensor
    tasks_data: torch.Tensor
    tasks_mask: torch.Tensor


class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict) -> None:
        super().__init__(observation_space, features_dim=VALUE_WIDTH)

    def forward(self, observation: Observation) -> Batch:
        num_satellites = observation["num_satellites"].argmax(-1)
        num_tasks = observation["num_tasks"].argmax(-1)
        time_step = observation["time_step"].argmax(-1)

        max_num_satellites = cast(int, num_satellites.max().item())
        max_num_tasks = cast(int, num_tasks.max().item())

        constellation_sensor_type = observation["constellation_sensor_type"].reshape(
            observation["constellation_sensor_type"].shape[0],
            MAX_NUM_SATELLITES,
            -1,
        ).argmax(-1)
        constellation_sensor_type = constellation_sensor_type[:, :max_num_satellites].int()
        constellation_sensor_enabled = observation["constellation_sensor_enabled"][:, :max_num_satellites].int()
        constellation_data = observation["constellation_data"][:, :max_num_satellites]

        constellation_mask = torch.zeros(
            [num_satellites.shape[0], max_num_satellites],
            dtype=torch.bool,
            device=constellation_data.device,
        )
        for index, value in enumerate(num_satellites):
            constellation_mask[index, : value] = True

        tasks_sensor_type = observation["tasks_sensor_type"].reshape(
            observation["tasks_sensor_type"].shape[0],
            MAX_NUM_TASKS,
            -1,
        ).argmax(-1)
        tasks_sensor_type = tasks_sensor_type[:, :max_num_tasks].int()
        tasks_data = observation["tasks_data"][:, :max_num_tasks]

        tasks_mask = torch.zeros(
            [num_tasks.shape[0], max_num_tasks],
            dtype=torch.bool,
            device=tasks_data.device,
        )
        for index, value in enumerate(num_tasks):
            tasks_mask[index, : value] = True

        return Batch(
            time_step=time_step,
            constellation_sensor_type=constellation_sensor_type,
            constellation_sensor_enabled=constellation_sensor_enabled,
            constellation_data=constellation_data,
            constellation_mask=constellation_mask,
            tasks_sensor_type=tasks_sensor_type,
            tasks_data=tasks_data,
            tasks_mask=tasks_mask,
        )


class CriticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._network = nn.Sequential(
            nn.Linear(1 + SATELLITE_DIM + TASK_DIM + 2, 256),
            nn.ReLU(),
            nn.Linear(256, VALUE_WIDTH),
            nn.ReLU(),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        time_step = batch.time_step.to(dtype=torch.float32).reshape(batch.time_step.shape[0], 1) / 3600.0
        constellation_mask = batch.constellation_mask.unsqueeze(-1)
        tasks_mask = batch.tasks_mask.unsqueeze(-1)
        constellation_mean = (
            (batch.constellation_data * constellation_mask).sum(dim=1)
            / constellation_mask.sum(dim=1).clamp_min(1)
        )
        tasks_mean = (
            (batch.tasks_data * tasks_mask).sum(dim=1)
            / tasks_mask.sum(dim=1).clamp_min(1)
        )
        counts = torch.stack(
            (
                batch.constellation_mask.sum(dim=1).to(dtype=torch.float32) / MAX_NUM_SATELLITES,
                batch.tasks_mask.sum(dim=1).to(dtype=torch.float32) / MAX_NUM_TASKS,
            ),
            dim=-1,
        )
        return self._network(torch.cat((time_step, constellation_mean, tasks_mean, counts), dim=-1))


class ActorCritic(nn.Module):
    def __init__(
        self,
        *,
        actor_config: AEOSFormerConfig,
    ) -> None:
        super().__init__()
        self.latent_dim_pi = MAX_NUM_TASKS
        self.latent_dim_vf = VALUE_WIDTH
        self.actor = build_actor(actor_config)
        self.critic = CriticModel()

    def forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(batch), self.forward_critic(batch)

    def forward_actor(self, batch: Batch) -> torch.Tensor:
        device = next(self.actor.parameters()).device
        batch = Batch(*[tensor.to(device) for tensor in batch])
        logits = self.actor.predict(*batch)
        padding = logits.new_full(
            (logits.shape[0], MAX_NUM_SATELLITES, MAX_NUM_TASKS),
            float("-inf"),
        )
        padding[..., 0] = 0
        padding[:, : logits.shape[1], : logits.shape[2]] = logits
        return padding

    def forward_critic(self, batch: Batch) -> torch.Tensor:
        device = next(self.critic.parameters()).device
        batch = Batch(*[tensor.to(device) for tensor in batch])
        return self.critic(batch)


class AEOSFormerPPOPolicy(ActorCriticPolicy):
    def __init__(
        self,
        *args,
        actor_config: AEOSFormerConfig,
        load_model_from: list[str] | None = None,
        **kwargs,
    ) -> None:
        self._actor_config = actor_config
        self._load_model_from = list(load_model_from or [])
        super().__init__(
            *args,
            ortho_init=False,
            features_extractor_class=FeatureExtractor,
            **kwargs,
        )
        if self._load_model_from:
            state_dict = normalized_state_dict(self._load_model_from[-1])
            self.mlp_extractor.actor.load_state_dict(state_dict, strict=False)  # type: ignore[arg-type]

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ActorCritic(actor_config=self._actor_config)  # type: ignore[assignment]

    def _get_action_dist_from_latent(
        self,
        latent_pi: torch.Tensor,
    ) -> Distribution:
        self.action_dist.distribution = [
            Categorical(logits=logits) for logits in latent_pi.unbind(1)
        ]
        return self.action_dist
