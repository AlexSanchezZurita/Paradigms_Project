import warnings
import gymnasium as gym
import os
import wandb
import torch
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

warnings.filterwarnings('ignore')

# Set your W&B API key here (replace with your API key)
os.environ["WANDB_API_KEY"] = "b03e9793c781306cd95072994865413eb3972360" 

wandb.init(
    project="pong", 
    entity="alexxsarda",  # Replace with your W&B username
    sync_tensorboard=True,  # Sync with TensorBoard logs
    monitor_gym=True,  # Track gym environment metrics
    save_code=True  # Save the script
)

# Parameters
MODEL_NAME = "ppo-v1"
ENV_NAME = "PongNoFrameskip-v4"
TOTAL_TIMESTEPS = 10_000_000  # Increased for better performance
SAVE_PATH = "./models/"
LOG_DIR = "./logs/"
NUM_ENVS = 8  # Use multiple environments for more stable training

# Create directories
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Custom callback for W&B logging
from stable_baselines3.common.callbacks import BaseCallback

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # If the environment info is available, log rewards
        # PPO doesn't directly provide "episode" info in locals by default,
        # but W&B will capture metrics from monitor logs.
        # If needed, we can extract more metrics here.
        return True

# Create the vectorized Atari environment
# This will handle standard Atari preprocessing internally:
# It uses NoopResetEnv, MaxAndSkipEnv, and relevant wrappers.
env = make_atari_env(ENV_NAME, n_envs=NUM_ENVS, wrapper_kwargs={
    "clip_reward": False,  # Match the training config
    "terminal_on_life_loss": True,
})
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

# Use known stable hyperparameters for Pong
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    device="cuda" if torch.cuda.is_available() else "cpu",
    # Following known good hyperparams for Pong
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
    name_prefix="ppo_checkpoint"
)

wandb_callback = WandbCallback()

print("Starting training on Pong with multiple parallel envs...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, wandb_callback])

model.save(SAVE_PATH + MODEL_NAME + "_final")
print(f"Model saved as {SAVE_PATH + MODEL_NAME}_final")
