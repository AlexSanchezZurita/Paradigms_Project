# Learning to Win: RL Strategies for Simple and Complex Games
# IMAGEN O VIDEO O GIF DE UNO DE LOS ENVIRONMENTS
## Overview
This repository contains the implementation and results of our project for the Artificial Intelligence course at [Insert Institution Name], under the guidance of Professor Jordi Casas. The project explores reinforcement learning (RL) strategies for solving challenges in simple and complex environments using the **Arcade Learning Environment (ALE)** and **PettingZoo**. We evaluate techniques ranging from basic **Deep Q-Networks (DQN)** to advanced methods like **Proximal Policy Optimization (PPO)**, alongside **Learning by Imitation**.

### Authors
- **Alex Roldan**
- **Alex Sanchez** 

### Project Date
December 10, 2024

---

## Contents
- **Introduction**  
  Overview of objectives, environments, and techniques.
  
- **Section 1: Simple ALE Environment (Enduro)**  
  - Environment setup, pre-processing, and model architecture.  
  - DQN enhancements and training insights.  

- **Section 2: Complex ALE Environment (Skiing)**  
  - Challenges with sparse rewards and delayed feedback.  
  - Transition from DQN to PPO.  
  - Reward shaping experiments and Learning by Imitation.

- **Section 3: Pong**  
  - Challenges in training agents for competitive Pong.  
  - Results for both left and right-side agents.  

- **Discussion and Insights**  
  Key takeaways and challenges encountered.

- **Conclusion**  
  Summary of findings and future directions.

---
## Setup Instructions

### Prerequisites
- Python 3.8+
- Libraries:
  - Stable-Baselines3
  - PettingZoo
  - OpenAI Gym
  - NumPy
  - Matplotlib

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/[YourGithubHandle]/[YourRepoName].git
   cd [YourRepoName]
  ```

2.Install required libraries
```bash
pip install -r requirements.txt
```

### Run the code

## Results and Observations

### Enduro (Simple Environment)
- **Model:** Enhanced DQN
- **Performance:** Achieved near-optimal performance after hyperparameter tuning and model upgrades. The agent consistently surpassed baseline policies.
# IMAGEN O VIDEO O GIF DEL ENVIRONMENT
### Skiing (Complex Environment)
- **Model:** PPO
- **Challenges:** Sparse rewards led to suboptimal strategies (e.g., skiing straight).
- **Solution:** Learning by Imitation improved policy initialization but required careful reward shaping.
# IMAGEN O VIDEO O GIF DEL ENVIRONMENT
### Pong
- **Model:** PPO
- **Findings:** Right-side agent overfitted to its training conditions, achieving a win rate of 100% against AI opponents. Left-side agent showed more natural movements but with lower performance due to preprocessing issues.
# IMAGEN O VIDEO O GIF DEL ENVIRONMENT
---

## Discussion and Insights
- **Challenges:** Sparse rewards and reliance on hyperparameter tuning were significant obstacles.
- **Insights:** Combining human demonstrations with reinforcement learning can accelerate training and improve outcomes.
- **Future Directions:**
  - Testing additional RL algorithms (e.g., Rainbow DQN).  
  - Developing multi-agent competitive scenarios.  
  - Automating hyperparameter optimization.

---

## References
- **Stable-Baselines3 Documentation**  
  https://stable-baselines3.readthedocs.io/
- **PettingZoo Library**  
  https://www.pettingzoo.ml/
- **OpenAI Gym**  
  https://gym.openai.com/

---

For further details, please refer to the full project report included in the repository.
