import math
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm

import isaaclab_experiments.go1_low_level.mdp.rewards as go1_mdp


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=)
    terminated = RewTerm(func=mdp.is_terminated, weight=)

    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=)
    dof_pos_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=)
    dof_torque_limit = RewTerm(func=mdp.applied_torque_limits, weight=)

    # footswing height tracking
    footswing = RewTerm(
        func=go1_mdp.FootClearenceReward,
        weight=,
        params={
            "target_height": 0.08,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=go1_mdp.FOOT_NAMES, preserve_order=True
            ),
        },
    )
