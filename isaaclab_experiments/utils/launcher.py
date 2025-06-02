import argparse
from packaging import version

from isaaclab.app import AppLauncher

class PlanningApp:

    def __init__(self):
        # parsing arguments
        parser = self.parse_args()

        # append AppLauncher cli args
        AppLauncher.add_app_launcher_args(parser)
        self.args_cli = parser.parse_args()
        # always enable cameras to record video
        if self.args_cli.video:
            self.args_cli.enable_cameras = True

        # launch omniverse app
        self.app_launcher = AppLauncher(self.args_cli)
        self.simulation_app = self.app_launcher.app
    
    def parse_args(self):
        # add argparse arguments
        parser = argparse.ArgumentParser(description="Play planning of an RL agent from skrl.")
        # simulation
        parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
        parser.add_argument("--task", type=str, default=None, help="Name of the task.")
        # policy
        parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
        parser.add_argument(
            "--use_pretrained_checkpoint",
            action="store_true",
            help="Use the pre-trained checkpoint from Nucleus.",
        )
        parser.add_argument(
            "--ml_framework",
            type=str,
            default="torch",
            choices=["torch", "jax", "jax-numpy"],
            help="The ML framework used for training the skrl agent.",
        )
        parser.add_argument(
            "--algorithm",
            type=str,
            default="PPO",
            choices=["AMP", "PPO", "IPPO", "MAPPO"],
            help="The RL algorithm used for training the skrl agent.",
        )
        parser.add_argument("--real-time", action="store_true", default=True, help="Run in real-time, if possible.")
        #video
        parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
        parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
        parser.add_argument(
            "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
        )
        return parser
    
    def check_skrl_version(self, skrl):
        SKRL_VERSION = "1.4.2"
        if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
            skrl.logger.error(
                f"Unsupported skrl version: {skrl.__version__}. "
                f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
            )
            exit()