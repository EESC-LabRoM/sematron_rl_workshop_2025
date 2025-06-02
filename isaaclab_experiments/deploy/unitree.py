# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import Optional
import torch
import numpy as np
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.sensors.physics import IMUSensor
from isaacsim.robot.policy.examples.controllers.config_loader import (
    get_robot_joint_properties,
)
from isaaclab.utils.types import ArticulationActions
from isaaclab_experiments.deploy.actuator_net import DEFAULT_POSE, ACTIONS_ORDERING
from isaaclab_experiments.go1_flat_deploy.cfg import GO1_ACTUATOR_CFG


class AnymalFlatTerrainPolicy(PolicyController):
    """The ANYmal Robot running a Flat Terrain Locomotion Policy"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "anymal",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize anymal robot, import policy and actuator network.
        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot
        """
        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(
            r"/home/nexus/Repositories/go1-flat-deploy/policy_jit.pt",
            r"/home/nexus/Repositories/go1-flat-deploy/logs/skrl/unitree_go1_flat/2025-03-20_17-20-08_ppo/params/env.yaml",
        )

        self.policy = self.policy.to("cpu")
        self._action_scale = 0.25
        self._previous_action = np.zeros(12)
        self._policy_counter = 0

        self.imu_path = self.robot.prim_path + "/trunk/imu_link"
        self._imu_sensor = IMUSensor(
            prim_path=self.imu_path + "/imu_sensor",
            name="imu",
            dt=0.005,
            translation=np.array([0, 0, 0]),
            orientation=np.array([1, 0, 0, 0]),
        )

    def _compute_observation(self, command):
        """
        Computes the the observation vector for the policy

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()

        pos_IB, q_IB = self.robot.get_world_pose()
        current_joint_pos = self.robot.get_joint_positions(
            joint_indices=self.joint_indexes
        )
        current_joint_vel = self.robot.get_joint_velocities(
            joint_indices=self.joint_indexes
        )

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.concatenate(
            [
                lin_vel_b,
                ang_vel_b,
                gravity_b,
                command,
                current_joint_pos - self.default_pos,
                current_joint_vel,
                self._previous_action,
            ]
        )
        return obs

    def forward(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        """
        # self._update_imu_sensor_data()

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            with torch.no_grad():
                self.action = self.policy(torch.tensor(obs, dtype=torch.float32))
            self._previous_action = np.array(self.action).copy()

        # The learning controller uses the order of
        current_joint_pos = self.robot.get_joint_positions(
            joint_indices=self.joint_indexes
        )
        current_joint_vel =self.robot.get_joint_velocities(
            joint_indices=self.joint_indexes
        )

        with torch.no_grad():
            joint_torques = self._actuator_network.compute(
                joint_pos=torch.tensor(current_joint_pos),
                joint_vel=torch.tensor(current_joint_vel),
                control_action=ArticulationActions(
                    joint_positions=self._action_scale * self.action + self.default_pos
                ),
            )

        self.robot.set_joint_efforts(
            efforts=np.array(joint_torques.joint_efforts),
            joint_indices=self.joint_indexes,
        )

        self._policy_counter += 1

    def initialize(self, physics_sim_view=None) -> None:
        """
        Initialize the articulation interface, set up drive mode
        """
        self.robot.initialize(physics_sim_view=physics_sim_view)

        self.robot.get_articulation_controller().set_effort_modes("force")
        self.robot.get_articulation_controller().switch_control_mode("effort")

        max_effort, max_vel, stiffness, damping, self.default_pos, self.default_vel = (
            get_robot_joint_properties(
                {"scene": self.policy_env_params}, self.robot.dof_names
            )
        )

        self.robot._articulation_view.set_gains(stiffness, damping)
        self.robot._articulation_view.set_max_efforts(max_effort)
        self.robot._articulation_view.set_max_joint_velocities(max_vel)

        self.default_pos = np.array([DEFAULT_POSE[i] for i in ACTIONS_ORDERING])

        self.robot.set_joints_default_state(
            positions=self.default_pos,
            velocities=np.zeros(12),
            efforts=np.zeros(12),
        )

        self.robot.set_joint_positions(self.default_pos)

        self.joint_indexes = np.array(
            [self.robot.get_dof_index(name) for name in ACTIONS_ORDERING]
        )

        self._actuator_network = GO1_ACTUATOR_CFG.class_type(GO1_ACTUATOR_CFG, 
                                                             num_envs=1, 
                                                             joint_names=ACTIONS_ORDERING, 
                                                             joint_ids=self.joint_indexes, 
                                                             device="cpu"
                                                             )