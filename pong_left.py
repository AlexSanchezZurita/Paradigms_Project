import warnings
import gymnasium as gym
import os
import wandb
import torch
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor
from stable_baselines3.common.env_util import make_atari_env
import numpy as np

warnings.filterwarnings('ignore')

# Set your W&B API key here (replace with your API key)
os.environ["WANDB_API_KEY"] = "b03e9793c781306cd95072994865413eb3972360" 

wandb.init(
    project="pong-left-side", 
    entity="alexxsarda",  # Replace with your W&B username
    sync_tensorboard=True,  # Sync with TensorBoard logs
    monitor_gym=True,  # Track gym environment metrics
    save_code=True  # Save the script
)

# Parameters
MODEL_NAME = "ppo-v1-left"
ENV_NAME = "PongNoFrameskip-v4"
TOTAL_TIMESTEPS = 1_500_000
SAVE_PATH = "./models/"
LOG_DIR = "./logs/"
NUM_ENVS = 8

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Custom callback for W&B logging
class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True

# Custom observation wrapper to flip the frames horizontally
class FlipObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Observation space remains the same shape
        self.observation_space = env.observation_space

    def observation(self, obs):
        # obs shape is typically (channels, height, width)
        # Flip horizontally along the width dimension
        return obs[..., ::-1]

# Create the vectorized Atari environment with standard Atari wrappers
env = make_atari_env(ENV_NAME, n_envs=NUM_ENVS, wrapper_kwargs={
    "clip_reward": False,
    "terminal_on_life_loss": True,
})

# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

# Apply the flipping wrapper at the vectorized level
# We wrap after frame stacking so the entire stacked observation is flipped
# VecMonitor is used to monitor returns and episodes
env = VecMonitor(env)

# For flipping, we need to apply at the individual env level.
# make_atari_env returns a DummyVecEnv or SubprocVecEnv. 
# To apply a custom wrapper to each environment, we can do so by specifying 
# a custom environment factory:
def flip_obs_factory():
    inner_env = gym.make(ENV_NAME)
    # Apply the same wrappers as make_atari_env does if needed, 
    # but since we are already using make_atari_env above, let's do a direct approach:
    # NOTE: Another approach is to integrate FlipObservationWrapper directly into make_atari_env's wrapper_kwargs,
    # but since it doesn't accept that directly, we can do the flipping after frame stacking.
    # Unfortunately, VecFrameStack can't be applied after a custom wrapper easily unless 
    # we do it differently:
    # 
    # Let's do this: 
    # 1. Create env with make_atari_env as before
    # 2. Wrap after creation by directly wrapping the VecEnv result.
    #
    # Since VecEnv doesn't directly allow a per-environment wrapper, 
    # we have to do the flipping at the end of the chain.
    #
    # We'll use a VecEnvWrapper for flipping.

    return inner_env

# Create a VecEnv that uses the flipping wrapper.
# Since we need to flip frames at the vectorized level and stable-baselines doesn't have a built-in 
# VecEnv wrapper for flipping, we can create a custom VecEnvWrapper:

from stable_baselines3.common.vec_env import VecEnvWrapper

import numpy as np

class FlipVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv, observation_space=venv.observation_space, action_space=venv.action_space)

    def reset(self):
        obs = self.venv.reset()
        # Flip horizontally and make a contiguous copy
        obs = np.flip(obs, axis=-1).copy(order='C')
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # Flip horizontally and make a contiguous copy
        obs = np.flip(obs, axis=-1).copy(order='C')
        return obs, rewards, dones, infos


# We need to recreate the env so we can insert the flipping wrapper after the frame stacking:
env = make_atari_env(ENV_NAME, n_envs=NUM_ENVS, wrapper_kwargs={
    "clip_reward": False,
    "terminal_on_life_loss": True,
})
env = VecFrameStack(env, n_stack=4)
env = VecMonitor(env)
env = FlipVecEnvWrapper(env)

# Define the model
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    device="cuda" if torch.cuda.is_available() else "cpu",
    learning_rate=2.5e-4,
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    clip_range=0.1,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={"normalize_images": False},
)

checkpoint_callback = CheckpointCallback(
    save_freq=500_000,
    save_path=SAVE_PATH,
    name_prefix="ppo_checkpoint_left"
)

wandb_callback = WandbCallback()

print("Starting training on Pong (flipped horizontally) with multiple parallel envs...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, wandb_callback])

model.save(SAVE_PATH + MODEL_NAME + "_final")
print(f"Model saved as {SAVE_PATH + MODEL_NAME}_final")
