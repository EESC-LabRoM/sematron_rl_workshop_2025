import numpy as np
import torch
import math
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.sim.spawners.meshes import MeshCuboidCfg
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
import weakref

DEBUG_VIS = False

FOOT_NAMES = [
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
]


GREEN_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": MeshCuboidCfg(
            size=(1.0, 0.02, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        )
    }
)
BLUE_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": MeshCuboidCfg(
            size=(1.0, 0.02, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        )
    }
)


def finer_track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, vel_limit: float, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    is_within_bounds = torch.linalg.vector_norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < vel_limit
    
    return torch.exp(-lin_vel_error / std**2) * is_within_bounds

def finer_track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, vel_limit: float, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    is_within_bounds = torch.abs(env.command_manager.get_command(command_name)[:, 2]) < vel_limit
    return torch.exp(-ang_vel_error / std**2) * is_within_bounds

class DesiredContactStates(ManagerTermBase):
    def __init__(
        self,
        cfg: RewTerm,
        env: ManagerBasedEnv,
    ) -> None:
        super().__init__(cfg, env)

        kappa_gait_probs: float = 0.07
        self.step_frequency_cmd: float = 3.5

        # Foot movement indices [0, 1)
        phase, offsets, bounds = cfg.params.get("gait")
        self.foot_indices = (
            torch.tensor([phase + offsets + bounds, offsets, bounds, phase])
            .remainder(
                torch.tensor(1.0),
            )
            .to(device=self._env.device)
        )

        # von mises distribution
        self.smoothing_cdf = torch.distributions.normal.Normal(0, kappa_gait_probs).cdf
        self.desired_contact_states = torch.zeros((4), device=self._env.device)
        self.dt = self._env.step_dt
        self.frequencies = torch.tensor(
            self.step_frequency_cmd, device=self._env.device
        )

    def __call__(self, env: ManagerBasedRLEnv, gait) -> torch.Tensor:
        # Remainder calculations and shifted indices
        self.foot_indices = torch.remainder(
            self.foot_indices + self.dt * self.frequencies, 1.0
        )

        foot_indices_shifted = self.foot_indices - 1
        foot_indices_half_shift = self.foot_indices - 0.5
        foot_indices_half_shift_shifted = foot_indices_half_shift - 1

        # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2
        self.desired_contact_states = self.smoothing_cdf(self.foot_indices) * (
            1 - self.smoothing_cdf(foot_indices_half_shift)
        ) + self.smoothing_cdf(foot_indices_shifted) * (
            1 - self.smoothing_cdf(foot_indices_half_shift_shifted)
        )
        return torch.cat(
            [
                torch.sin(2 * np.pi * self.foot_indices),
                torch.cos(2 * np.pi * self.foot_indices),
            ]
        ).repeat(env.num_envs, 1)


class FootClearenceReward(ManagerTermBase):
    asset_cfg: SceneEntityCfg
    target_height: float
    _debug_vis_handle = None

    def __init__(self, cfg: ObsTerm, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.desired_contact_caller: DesiredContactStates = (
            self._env.observation_manager.cfg.policy.clock.func
        )

        if "asset_cfg" not in cfg.params:
            raise ValueError("asset_cfg missing from FootClearenceReward configuration")
        self.asset_cfg = self.cfg.params.get("asset_cfg")
        self.asset: RigidObject = env.scene[self.asset_cfg.name]

        if "target_height" not in cfg.params:
            raise ValueError(
                "target_height missing from FootClearenceReward configuration"
            )
        self.target_height = self.cfg.params.get("target_height")
        self.target_heights = torch.zeros_like(self.desired_contact_caller.foot_indices)
        self.set_debug_vis(DEBUG_VIS)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        if self._debug_vis_handle:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    def __call__(
        self,
        env,
        target_height,
        asset_cfg,
    ) -> torch.Tensor:
        """Reward the swinging feet for clearing a specified height off the ground"""
        foot_indices = self.desired_contact_caller.foot_indices
        desired_contact_states = self.desired_contact_caller.desired_contact_states

        self.target_heights = FootClearenceReward.compute_target_height(
            target_height=target_height, foot_indices=foot_indices
        )

        foot_z_target_error = torch.square(
            self.asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - self.target_heights
        ) 

        velocity = env.command_manager.get_command("base_velocity")
        reward = foot_z_target_error * (1 - desired_contact_states) * torch.linalg.norm(velocity, dim=1, keepdim=True)
        reward = torch.sum(reward, dim=1) 

        return reward

    @staticmethod
    def compute_target_height(target_height, foot_indices):
        phases = 1 - torch.abs(
            1.0 - torch.clip((foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0
        )
        return target_height * phases + 0.02  # offset for foot radius 2cm

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the command data.

        Args:
            debug_vis: Whether to visualize the command data.

        Returns:
            Whether the debug visualization was successfully set. False if the command
            generator does not support debug visualization.
        """
        # toggle debug visualization objects
        self._set_debug_vis_impl(debug_vis)

        # toggle debug visualization handles
        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(
                        event
                    )
                )
        else:
            # remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None

        # return success
        return True

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_ve_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(
                    GREEN_ARROW_X_MARKER_CFG.replace(
                        prim_path="/Visuals/Command/foot_clearence_goal"
                    )
                )
                self.current_vel_visualizer = VisualizationMarkers(
                    BLUE_ARROW_X_MARKER_CFG.replace(
                        prim_path="/Visuals/Command/foot_clearence_current"
                    )
                )
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data

        if not self.asset.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = (
            self.asset.data.body_pos_w[:, self.asset_cfg.body_ids].squeeze(0).clone()
        )
        base_pos_w[:, 2] = 0.0
        num_joints = base_pos_w.shape[0]

        target_heights = (
            self.target_heights.repeat(self._env.num_envs, 1).squeeze(0) - 0.02
        )
        current_heights = (
            self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, 2].squeeze(0) - 0.02
        )

        # display markers
        arrow_scale = torch.tensor([1.0, 1.0, 1.0])
        target_scales = (
            arrow_scale.to(device=self._env.device).repeat(num_joints, 1).clone()
        )
        current_scales = (
            arrow_scale.to(device=self._env.device).repeat(num_joints, 1).clone()
        )
        target_scales[:, 0] = target_heights
        current_scales[:, 0] = current_heights

        current_pose_w_offset = base_pos_w.clone()
        current_pose_w_offset[:, 2] = current_heights / 2
        target_pose_w_offset = base_pos_w.clone()
        target_pose_w_offset[:, 2] = target_heights / 2

        self.goal_vel_visualizer.visualize(
            target_pose_w_offset + torch.tensor([0.02, 0, 0], device=self._env.device),
            torch.tensor([0.707, 0, -0.707, 0]).repeat(num_joints, 1),
            target_scales,
        )

        self.current_vel_visualizer.visualize(
            current_pose_w_offset - torch.tensor([0.02, 0, 0], device=self._env.device),
            torch.tensor([0.707, 0, -0.707, 0]).repeat(num_joints, 1),
            current_scales,
        )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    fine_track_lin_vel_xy_exp = RewTerm(
        func=finer_track_lin_vel_xy_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.1), "vel_limit": 0.2},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    fine_track_ang_vel_z_exp = RewTerm(
        func=finer_track_ang_vel_z_exp, weight=0.35, params={"command_name": "base_velocity", "std": math.sqrt(0.1), "vel_limit": 0.2},
    )
    
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    terminated = RewTerm(func=mdp.is_terminated, weight=-5)

    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1)
    dof_pos_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05)
    dof_torque_limit = RewTerm(func=mdp.applied_torque_limits, weight=-1.0)

    # footswing height tracking
    footswing = RewTerm(
        func=FootClearenceReward,
        weight=-200,
        params={
            "target_height": 0.08,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=FOOT_NAMES, preserve_order=True
            ),
        },
    )