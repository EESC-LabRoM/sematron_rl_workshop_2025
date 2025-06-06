import torch

def update_camera(env):
    robot = env.scene["robot"]
    root_pos = robot.data.root_state_w[:, 0:3]  # Shape: (num_envs, 3)
    robot_pos = root_pos[0]  # Assuming num_envs = 1 or you're visualizing the first env

    # Define camera offset (relative to world coordinates)
    camera_offset = torch.tensor([10.0, 0.0, 10.0], device=robot_pos.device)
    look_at_offset = torch.tensor([0.0, 0.0, 1.0], device=robot_pos.device)

    # Compute camera world position and look-at
    camera_pos = robot_pos + camera_offset
    camera_look = robot_pos + look_at_offset

    # Set the camera view
    env.sim.set_camera_view(eye=camera_pos.cpu().numpy(), target=camera_look.cpu().numpy())