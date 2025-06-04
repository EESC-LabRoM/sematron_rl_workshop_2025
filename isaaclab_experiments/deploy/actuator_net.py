import argparse
import json
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence

import torch
import numpy as np

DEFAULT_POSE = {
    "FL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RL_hip_joint": 0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
}

# Actions ordering:
ACTIONS_ORDERING = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]


@dataclass
class ActuatorNetMLPCfg:
    """Configuration for MLP-based actuator model."""

    network_file: str
    """Path to the file containing network weights."""
    pos_scale: float
    """Scaling of the joint position errors input to the network."""
    vel_scale: float
    """Scaling of the joint velocities input to the network."""
    torque_scale: float
    """Scaling of the joint efforts output from the network."""
    input_order: Literal["pos_vel", "vel_pos"]
    """Order of the inputs to the network.

    The order can be one of the following:

    * ``"pos_vel"``: joint position errors followed by joint velocities
    * ``"vel_pos"``: joint velocities followed by joint position errors
    """
    input_idx: Iterable[int]
    """
    Indices of the actuator history buffer passed as inputs to the network.

    The index *0* corresponds to current time-step, while *n* corresponds to n-th
    time-step in the past. The allocated history length is `max(input_idx) + 1`.
    """


class ActuatorNetMLP:
    """Actuator model based on multi-layer perceptron and joint history.

    Many times the analytical model is not sufficient to capture the actuator dynamics, the
    delay in the actuator response, or the non-linearities in the actuator. In these cases,
    a neural network model can be used to approximate the actuator dynamics. This model is
    trained using data collected from the physical actuator and maps the joint state and the
    desired joint command to the produced torque by the actuator.

    This class implements the learned model as a neural network based on the work from
    :cite:t:`hwangbo2019learning`. The class stores the history of the joint positions errors
    and velocities which are used to provide input to the neural network. The model is loaded
    as a TorchScript.

    Note:
        Only the desired joint positions are used as inputs to the network.

    """

    cfg: ActuatorNetMLPCfg
    """The configuration of the actuator model."""

    def __init__(
        self, cfg: ActuatorNetMLPCfg, joint_names: list[str], device: str = "cpu"
    ):
        self.cfg = cfg
        self.joint_names = joint_names
        self.num_joints = len(joint_names)
        self.device = device

        # load the model from JIT file
        self.network = torch.jit.load(self.cfg.network_file, map_location=self.device)

        # create buffers for MLP history
        history_length = max(self.cfg.input_idx) + 1

        self._joint_pos_error_history = torch.zeros(
            1, history_length, self.num_joints, device=self.device
        )
        self._joint_vel_history = torch.zeros(
            1, history_length, self.num_joints, device=self.device
        )

        self._joint_vel = torch.zeros(1, self.num_joints, device=self.device)

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int]):
        # reset the history for the specified environments
        self._joint_pos_error_history[env_ids] = 0.0
        self._joint_vel_history[env_ids] = 0.0

    def compute_torques(
        self,
        control_positions: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        # move history queue by 1 and update top of history
        # -- positions
        self._joint_pos_error_history = self._joint_pos_error_history.roll(1, 1)
        self._joint_pos_error_history[:, 0] = torch.tensor(
            control_positions - joint_pos
        )
        # -- velocity
        self._joint_vel_history = self._joint_vel_history.roll(1, 1)
        self._joint_vel_history[:, 0] = torch.tensor(joint_vel)
        # save current joint vel for dc-motor clipping
        self._joint_vel[:] = torch.tensor(joint_vel)

        # compute network inputs
        # -- positions
        pos_input = torch.cat(
            [
                self._joint_pos_error_history[:, i].unsqueeze(2)
                for i in self.cfg.input_idx
            ],
            dim=2,
        )
        pos_input = pos_input.view(self.num_joints, -1)
        # -- velocity
        vel_input = torch.cat(
            [self._joint_vel_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
            dim=2,
        )
        vel_input = vel_input.view(self.num_joints, -1)
        # -- scale and concatenate inputs
        if self.cfg.input_order == "pos_vel":
            network_input = torch.cat(
                [pos_input * self.cfg.pos_scale, vel_input * self.cfg.vel_scale], dim=1
            )
        elif self.cfg.input_order == "vel_pos":
            network_input = torch.cat(
                [vel_input * self.cfg.vel_scale, pos_input * self.cfg.pos_scale], dim=1
            )
        else:
            raise ValueError(
                f"Invalid input order for MLP actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
            )

        # run network inference
        with torch.no_grad():
            torques = self.network(network_input).view(1, self.num_joints)

        self.computed_effort = torques.view(1, self.num_joints) * self.cfg.torque_scale

        return self.computed_effort

