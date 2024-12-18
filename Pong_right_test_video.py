import os
import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.env_util import make_atari_env

# Paths (must match those used in training)
MODEL_NAME = "ppo-v1"
ENV_NAME = "PongNoFrameskip-v4"
SAVE_PATH = "./models/"

# Ensure the videos folder exists
os.makedirs("videos", exist_ok=True)

# Create the same environment as in training
eval_env = make_atari_env(
    ENV_NAME, 
    n_envs=1, 
    wrapper_kwargs={
        "clip_reward": False,
        "terminal_on_life_loss": True
    }
)
eval_env = VecFrameStack(eval_env, n_stack=4)

# Wrap the environment to record videos
# We'll record the very first episode (record_video_trigger=lambda x: x == 0)
eval_env = VecVideoRecorder(
    eval_env, 
    video_folder="videos", 
    record_video_trigger=lambda episode_id: episode_id == 0, 
    video_length=1000  # number of steps to record
)

# Load the trained model
model = PPO.load("models\pong_RIGHT.zip")

# Reset environment
obs = eval_env.reset()

# Run a single episode and let the agent play
for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = eval_env.step(action)
    # When a single environment done is True, you may want to break or handle resets
    if dones[0]:
        break

# Close the environment after recording
eval_env.close()

print("Video recording completed! Check the 'videos' folder for the output file.")
