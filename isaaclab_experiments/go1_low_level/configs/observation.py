from isaaclab_experiments.go1_low_level.mdp.rewards import DesiredContactStates
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import ObservationGroupCfg as ObsGroup

gaits = {
    "pronking": [0, 0, 0],
    "trotting": [0.5, 0, 0],
    "bounding": [0, 0.5, 0],
    "pacing": [0, 0, 0.5],
}

GAIT = gaits["trotting"]


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.02, n_max=0.02)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05
        )
        actions = ObsTerm(func=mdp.last_action, noise=Unoise(n_min=-0.01, n_max=0.01))
        clock = ObsTerm(
            func=DesiredContactStates,
            params={
                "gait": GAIT,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
