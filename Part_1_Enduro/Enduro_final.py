import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import gymnasium 
import collections
from collections import namedtuple, deque
from copy import deepcopy
import os
import ale_py
import cv2
import wandb
import random
import numpy as np


# put wandb api key here
wandb.login(key="b03e9793c781306cd95072994865413eb3972360")

# Hyperparameters
lr = 0.001            # Learning rate
MEMORY_SIZE = 150000  # Maximum buffer capacity
MAX_EPISODES = 6000   # Maximum number of episodes (the agent must learn before reaching this value)
EPSILON = 1           # Initial value of epsilon
EPSILON_DECAY = .999  # Epsilon decay
GAMMA = 0.98          # Gamma value of the Bellman equation
BATCH_SIZE = 128       # Number of elements to extract from the buffer
BURN_IN = 1000        # Number of initial episodes used to fill the buffer before training
DNN_UPD = 4           # Neural network update rate
DNN_SYNC = 1000       # Frequency of synchronization between the neural network and the target network
N_STEP = 2            # Number of steps for N-step return


wandb.init(project="bowling", entity="alexxsarda", config={
    "learning_rate": lr,
    "memory_size": MEMORY_SIZE,
    "gamma": GAMMA,
    "batch_size": BATCH_SIZE,
    "epsilon_decay": EPSILON_DECAY,
    "max_episodes": MAX_EPISODES,
    "n_step": N_STEP
})

# check if cuda available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

ENV_NAME = "ALE/Enduro-v5"

class MaxAndSkipEnv(gymnasium.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False  # Updated to handle the 'truncated' flag in Gymnasium
        info = {}

        for _ in range(self._skip):
            obs, reward, done, truncated, step_info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward

            # Merge the info dictionaries
            info.update(step_info)

            if done or truncated:
                break

        # Take the maximum frame over the buffered observations
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, truncated, info

    def reset(self, *, seed=None, options=None):
        self._obs_buffer.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs_buffer.append(obs)
        return obs, info


class ProcessFrame84(gymnasium.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class BufferWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gymnasium.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0),
            dtype=dtype
        )

    # Update reset to accept additional arguments
    def reset(self, *, seed=None, options=None):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        observation, info = self.env.reset(seed=seed, options=options)
        return self.observation(observation), info

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], 
                                old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gymnasium.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_env(env_name):
    env = gymnasium.make(env_name, render_mode="rgb_array")
    print("Standard Env.        : {}".format(env.observation_space.shape))
    env = MaxAndSkipEnv(env)
    print("MaxAndSkipEnv        : {}".format(env.observation_space.shape))
    env = ProcessFrame84(env)
    print("ProcessFrame84       : {}".format(env.observation_space.shape))
    env = ImageToPyTorch(env)
    print("ImageToPyTorch       : {}".format(env.observation_space.shape))
    env = BufferWrapper(env, 4)
    print("BufferWrapper        : {}".format(env.observation_space.shape))
    env = ScaledFloatFrame(env)
    print("ScaledFloatFrame     : {}".format(env.observation_space.shape))
    
    return env

env = make_env(ENV_NAME)

class PrioritizedReplayBuffer:
    def __init__(self, memory_size=100000, alpha=0.7, beta=0.4, beta_increment=1e-4, epsilon=1e-5, n_step=2, gamma=0.99):
        self.memory_size = memory_size
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = epsilon  # Small value to ensure non-zero priority
        self.gamma = gamma
        self.n_step = n_step

        self.priorities = deque(maxlen=memory_size)
        self.replay_memory = deque(maxlen=memory_size)
        self.n_step_buffer = deque(maxlen=n_step)

        self.buffer = namedtuple('Buffer', field_names=['state', 'action', 'reward', 'done', 'next_state'])

    def append(self, state, action, reward, done, next_state):
        self.n_step_buffer.append((state, action, reward, done, next_state))

        # If the buffer contains enough elements for N-step
        if len(self.n_step_buffer) == self.n_step:
            n_state, n_action = self.n_step_buffer[0][:2]
            n_reward, n_next_state, n_done = self._get_n_step_info()
            self.replay_memory.append(self.buffer(n_state, n_action, n_reward, n_done, n_next_state))
            self.priorities.append(max(self.priorities, default=1))  # New experiences have max priority

        if done:
            self.n_step_buffer.clear()

    def _get_n_step_info(self):
        """Calculate N-step reward and next state."""
        reward, next_state, done = 0, self.n_step_buffer[-1][-1], self.n_step_buffer[-1][-2]
        for idx, (_, _, r, d, s_next) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** idx) * r
            if d:
                done = True
                next_state = s_next
                break
        return reward, next_state, done

    def sample_batch(self, batch_size=32):
        # Compute priorities as probabilities
        priorities = np.array(self.priorities, dtype=np.float32) ** self.alpha
        probabilities = priorities / np.sum(priorities)

        # Sample indices based on priorities
        indices = np.random.choice(len(self.replay_memory), batch_size, p=probabilities)
        samples = [self.replay_memory[idx] for idx in indices]

        # Compute importance sampling weights
        total_samples = len(self.replay_memory)
        weights = (total_samples * probabilities[indices]) ** -self.beta
        weights = weights / weights.max()  # Normalize for stability

        self.beta = min(1.0, self.beta + self.beta_increment)  # Increment beta toward 1

        batch = zip(*samples)
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD-errors."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + self.epsilon)

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.memory_size


class DuellingDQN(nn.Module):
    def __init__(self, env, learning_rate=1e-3, device='cpu'):
        super(DuellingDQN, self).__init__()
        self.device = device
        self.n_outputs = env.action_space.n
        self.learning_rate = learning_rate

        # Define the model
        self.model = nn.Sequential(
            # Shared convolutional layers
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Define the separate streams for state-value and action-advantage
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Single output for V(s)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_outputs)  # Outputs for A(s, a)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(device)

    def forward(self, state):
        x = self.model(state)  # Shared feature extraction
        x = x.view(x.size(0), -1)  # Flatten the feature maps

        # Compute state-value and action-advantage
        value = self.value_stream(x)  # Shape: (batch_size, 1)
        advantage = self.advantage_stream(x)  # Shape: (batch_size, n_actions)

        # Combine them to compute Q-values
        qvals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return qvals

    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.n_outputs)
        else:
            qvals = self.get_qvals(state)
            action = torch.argmax(qvals).item()
        return action

    def get_qvals(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Add batch dimension
        return self.forward(state_t)


class DQNAgent:
    def __init__(self, env, dnnetwork, buffer, epsilon, eps_decay, batch_size, n_step):
        self.env = env
        self.dnnetwork = dnnetwork
        self.target_network = deepcopy(dnnetwork).to(device)
        self.buffer = buffer
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.nblock = 100
        self.reward_threshold = self.env.spec.reward_threshold if self.env.spec.reward_threshold is not None else float('inf')
        self.initialize()
        self.all_losses = []
        self.n_step = n_step

        # scheduler that reduces learning rate when plateau is detected.
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.dnnetwork.optimizer, mode='max', factor=0.1, patience=10)


    def initialize(self):
        self.update_loss = []
        self.training_rewards = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.total_reward = 0
        self.step_count = 0
        self.state0 = self.env.reset()[0]

    def take_step(self, eps, mode='train'):
        if mode == 'explore':
            action = self.env.action_space.sample()
        else:
            action = self.dnnetwork.get_action(self.state0, eps)
            self.step_count += 1

        new_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.total_reward += reward
        self.buffer.append(self.state0, action, reward, done, new_state)
        self.state0 = new_state.copy()

        if done:
            self.state0 = self.env.reset()[0]

        return done


    def train(self, gamma=0.99, max_episodes=50000, batch_size=128, dnn_update_frequency=4, dnn_sync_frequency=2000):
        self.gamma = gamma
        video_dir = "videos"  # Directory to store videos
        os.makedirs(video_dir, exist_ok=True)  # Ensure the directory exists

        # Burn-in phase
        print("Filling replay buffer...")
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(self.epsilon, mode='explore')

        episode = 0
        training = True
        print("Training...")
        while training:
            self.state0 = self.env.reset()[0]
            self.total_reward = 0
            gamedone = False
            episode_frames = []  # To store frames for video logging

            while not gamedone:
                gamedone = self.take_step(self.epsilon, mode='train')

                # Capture frames for the current episode
                frame = self.env.render()
                episode_frames.append(frame)

                if self.step_count % dnn_update_frequency == 0:
                    self.update()

                if self.step_count % dnn_sync_frequency == 0:
                    self.target_network.load_state_dict(self.dnnetwork.state_dict())
                    self.sync_eps.append(episode)

                if gamedone:
                    episode += 1
                    self.training_rewards.append(self.total_reward)
                    self.update_loss = []
                    mean_rewards = np.mean(self.training_rewards[-self.nblock:])
                    self.mean_training_rewards.append(mean_rewards)
                    print(f"\rEpisode {episode} Mean Rewards {mean_rewards:.2f} Epsilon {self.epsilon}\t\t", end="")

                    # Update the Learning Rate Scheduler
                    self.scheduler.step(mean_rewards)

                    # Log the metrics
                    wandb.log({
                        "episode": episode,
                        "reward": self.total_reward,
                        "mean_reward": mean_rewards,
                        "epsilon": self.epsilon,
                    })

                    # Save video every 100 episodes
                    if episode % 600 == 0:
                        video_filename = os.path.join(video_dir, f"Enduro_{episode}.mp4")
                        self.save_video(episode_frames, video_filename)

                    if episode >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break

                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print(f'\nEnvironment solved in {episode} episodes!')
                        break

                    self.epsilon = max(self.epsilon * self.eps_decay, 0.01)

    def save_video(self, frames, filename, fps=30):
        """Save a list of frames as a video."""
        if len(frames) == 0:
            print("No frames to save.")
            return

        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        video = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        for frame in frames:
            video.write(frame)

        video.release()
        print(f"Saved video: {filename}")

    def calculate_loss(self, batch, indices, weights):
        states, actions, rewards, dones, next_states = [np.array(i) for i in batch]
        states_t = torch.FloatTensor(states).to(self.dnnetwork.device)
        next_states_t = torch.FloatTensor(next_states).to(self.dnnetwork.device)
        rewards_t = torch.FloatTensor(rewards).reshape(-1, 1).to(self.dnnetwork.device)
        actions_t = torch.LongTensor(actions).to(self.dnnetwork.device)
        dones_t = torch.BoolTensor(dones).to(self.dnnetwork.device)

        qvals = self.dnnetwork(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_actions = torch.argmax(self.dnnetwork(next_states_t), dim=-1)
            qvals_next = self.target_network(next_states_t).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            qvals_next[dones_t] = 0
            expected_qvals = rewards_t.squeeze(-1) + self.gamma ** self.n_step * qvals_next

        td_errors = expected_qvals - qvals
        loss = (torch.FloatTensor(weights).to(self.dnnetwork.device) * td_errors.pow(2)).mean()

        return loss, td_errors.detach().cpu().numpy()


    def update(self):
        self.dnnetwork.optimizer.zero_grad()
        batch, indices, weights = self.buffer.sample_batch(batch_size=self.batch_size)
        loss, td_errors = self.calculate_loss(batch, indices, weights)
        loss.backward()
        self.dnnetwork.optimizer.step()
        self.buffer.update_priorities(indices, td_errors)

        self.update_loss.append(loss.detach().cpu().numpy())
        self.all_losses.append(loss.detach().cpu().numpy())

# Initialize components and train
dqn_network = DuellingDQN(env, learning_rate=1e-3, device=device)
buffer = PrioritizedReplayBuffer(memory_size=50000, alpha=0.7, beta=0.4, n_step=N_STEP, gamma=GAMMA)
agent = DQNAgent(env, dqn_network, buffer, epsilon=EPSILON, eps_decay=EPSILON_DECAY, batch_size=BATCH_SIZE, n_step=N_STEP)
agent.train(max_episodes=MAX_EPISODES)
env.close()

torch.save(agent.dnnetwork.state_dict(), "Enduro.pth")
