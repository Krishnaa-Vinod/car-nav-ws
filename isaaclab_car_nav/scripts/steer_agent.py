# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with a constant steering agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Constant steering agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=300,
    help="Maximum number of environment steps to run before exiting.",
)
parser.add_argument(
    "--steer",
    type=float,
    default=0.25,
    help="Constant steering command value applied to the first two action dimensions.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_car_nav.tasks  # noqa: F401


def main():
    """Constant steering agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    env.reset()

    step_count = 0

    try:
        while simulation_app.is_running() and step_count < args_cli.max_steps:
            with torch.inference_mode():
                # steer only the first two action dimensions; keep other action terms zero.
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                if actions.shape[-1] < 2:
                    raise RuntimeError(
                        f"Expected at least 2 action dimensions for steering, got {actions.shape[-1]}."
                    )
                actions[:, 0:2] = args_cli.steer
                env.step(actions)
                step_count += 1

                if step_count % 50 == 0 or step_count == 1:
                    print(
                        f"[INFO]: Step {step_count}/{args_cli.max_steps} "
                        f"with constant action value {args_cli.steer:.3f}"
                    )

        print(f"[INFO]: Finished steering rollout after {step_count} steps.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
