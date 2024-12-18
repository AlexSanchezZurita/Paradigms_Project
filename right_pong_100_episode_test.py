import os
import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import matplotlib.pyplot as plt

# Configuration
MODEL_NAME = "ppo-v1"
ENV_NAME = "PongNoFrameskip-v4"
SAVE_PATH = "./models/"
NUM_EPISODES = 100

# Create the environment (same as in training)
eval_env = make_atari_env(
    ENV_NAME, 
    n_envs=1, 
    wrapper_kwargs={
        "clip_reward": False,
        "terminal_on_life_loss": True
    }
)
eval_env = VecFrameStack(eval_env, n_stack=4)

# Load the trained model
model = PPO.load(os.path.join(SAVE_PATH, "pong_RIGHT.zip"))

episode_rewards = []
wins = 0

cumulative_avg_rewards = []
cumulative_win_rates = []

# Run 100 test episodes
for episode in range(NUM_EPISODES):
    obs = eval_env.reset()
    done = False
    ep_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        ep_reward += rewards[0]
        done = dones[0]

    episode_rewards.append(ep_reward)
    if ep_reward > 0:
        wins += 1
    
    # Compute cumulative averages and win rate
    current_avg_reward = sum(episode_rewards) / (episode + 1)
    current_win_rate = wins / (episode + 1)
    cumulative_avg_rewards.append(current_avg_reward)
    cumulative_win_rates.append(current_win_rate)

# Close the environment after testing
eval_env.close()

# Final results
final_average_reward = sum(episode_rewards) / NUM_EPISODES
final_win_rate = wins / NUM_EPISODES

print("Testing completed!")
print(f"Number of episodes: {NUM_EPISODES}")
print(f"Average Reward: {final_average_reward}")
print(f"Win Rate: {final_win_rate * 100:.2f}%")

# Plotting the results
# Plot cumulative average reward over episodes
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPISODES + 1), cumulative_avg_rewards, label='Average Reward')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Cumulative Average Reward Over Episodes')
plt.grid(True)
plt.legend()
plt.show()

# Plot cumulative win rate over episodes
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPISODES + 1), cumulative_win_rates, label='Win Rate')
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.title('Cumulative Win Rate Over Episodes')
plt.grid(True)
plt.legend()
plt.show()
