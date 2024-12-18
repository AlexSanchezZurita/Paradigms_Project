import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import ale_py
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, VecEnvWrapper, VecVideoRecorder
from stable_baselines3.common.env_util import make_atari_env
import torch

# Configuration
MODEL_NAME = "pong_LEFT"  # Adjust based on your saved model's exact name
ENV_NAME = "PongNoFrameskip-v4"
SAVE_PATH = "./models/"
VIDEO_FOLDER = "./videos/"
NUM_ENVS = 1  # Single environment for recording
VIDEO_EPISODES = 1  # Number of episodes to record
TEST_EPISODES = 100  # Number of episodes for testing
SEED = 42  # Fixed seed for reproducibility

# Ensure the videos folder exists
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Custom VecEnv Wrapper to Flip Observations Horizontally
class FlipVecEnvWrapper(VecEnvWrapper):
    """
    A VecEnv wrapper that flips observations horizontally.
    """
    def __init__(self, venv):
        super().__init__(venv, observation_space=venv.observation_space, action_space=venv.action_space)

    def reset(self):
        obs = self.venv.reset()
        # Flip horizontally (assuming last axis is width)
        obs = np.flip(obs, axis=-1).copy(order='C')
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # Flip observations horizontally
        obs = np.flip(obs, axis=-1).copy(order='C')
        return obs, rewards, dones, infos

# Unified Environment Creation Function
def create_env(env_name, n_envs=1):
    """
    Creates a vectorized Atari environment with observation flipping.
    """
    env = make_atari_env(
        env_name,
        n_envs=n_envs,
        wrapper_kwargs={
            "clip_reward": False,
            "terminal_on_life_loss": True
        },
    )
    env = VecFrameStack(env, n_stack=4)
    env = VecMonitor(env)
    env = FlipVecEnvWrapper(env)
    return env

# Initialize the evaluation environment for testing
test_env = create_env(ENV_NAME, n_envs=NUM_ENVS)
test_env.seed(SEED)

# Load the trained left-side model
model_path = os.path.join(SAVE_PATH, MODEL_NAME)
model_file = model_path + ".zip"

if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file not found at {model_file}")

# Load the model with the evaluation environment
model = PPO.load(model_path, env=test_env, device="cuda" if torch.cuda.is_available() else "cpu")

# Testing over 100 episodes
episode_rewards = []
wins = 0

cumulative_avg_rewards = []
cumulative_win_rates = []

print("Starting 100-episode testing...")

for episode in range(1, TEST_EPISODES + 1):
    obs = test_env.reset()
    done = False
    ep_reward = 0.0
    step = 0

    while not done and step < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = test_env.step(action)
        ep_reward += rewards[0]
        done = dones[0]
        step += 1

    episode_rewards.append(ep_reward)
    if ep_reward > 0:
        wins += 1

    # Calculate cumulative average reward and win rate
    current_avg_reward = np.mean(episode_rewards)
    current_win_rate = wins / episode
    cumulative_avg_rewards.append(current_avg_reward)
    cumulative_win_rates.append(current_win_rate)

    print(f"Episode {episode}: Reward = {ep_reward}, Cumulative Avg Reward = {current_avg_reward:.2f}, Win Rate = {current_win_rate * 100:.2f}%")

# Final testing results
final_average_reward = np.mean(episode_rewards)
final_win_rate = wins / TEST_EPISODES

print("\nTesting completed!")
print(f"Number of episodes: {TEST_EPISODES}")
print(f"Average Reward: {final_average_reward:.2f}")
print(f"Win Rate: {final_win_rate * 100:.2f}%")

# Plotting the results
# Plot cumulative average reward over episodes
plt.figure(figsize=(12, 6))
plt.plot(range(1, TEST_EPISODES + 1), cumulative_avg_rewards, label='Cumulative Average Reward', color='blue')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Cumulative Average Reward Over Episodes')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot cumulative win rate over episodes
plt.figure(figsize=(12, 6))
plt.plot(range(1, TEST_EPISODES + 1), np.array(cumulative_win_rates) * 100, label='Cumulative Win Rate', color='green')
plt.xlabel('Episode')
plt.ylabel('Win Rate (%)')
plt.title('Cumulative Win Rate Over Episodes')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Now, set up the environment for video recording
video_env = create_env(ENV_NAME, n_envs=NUM_ENVS)
video_env.seed(SEED)

# Apply VecVideoRecorder to record a single episode
video_env = VecVideoRecorder(
    video_env,
    video_folder=VIDEO_FOLDER,
    record_video_trigger=lambda episode_id: episode_id < VIDEO_EPISODES,
    video_length=1000,  # Adjust as needed
    name_prefix="left_side_agent_playing"
)

# Load the model with the video environment
model = PPO.load(model_path, env=video_env, device="cuda" if torch.cuda.is_available() else "cpu")

# Reset the environment
obs = video_env.reset()

# Run a single episode and record the video
done = False
ep_reward = 0.0
step = 0

print("\nStarting video recording...")

while not done and step < 1000:
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = video_env.step(action)
    ep_reward += rewards[0]
    done = dones[0]
    step += 1

# Close the environment after recording
video_env.close()

print("Video recording completed! Check the 'videos' folder for the output file.")
print(f"Recorded Episode Reward: {ep_reward:.2f}")
