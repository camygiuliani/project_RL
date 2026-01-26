# Project for RL exam 2026
<!-- Badges -->
[![Python](https://img.shields.io/badge/python-3.12.3-blue)](https://www.python.org/)
[![Camilla Giuliani on GitHub](https://img.shields.io/badge/Camilla–Giuliani–GitHub-181717?style=plastic&logo=github)](https://github.com/camygiuliani)
[![Pietro D'Annibale on GitHub](https://img.shields.io/badge/Pietro–D%E2%80%99Annibale–GitHub-181717?style=plastic&logo=github)](https://github.com/Sassotek)
<!--........-->

Camilla Giuliani 1883207 &&  Pietro D'Annibale 1917211

## How to run the code
### 1. Clone the repository
```bash
git clone https://github.com/USERNAME/project_RL.git
cd project_RL
```
### 2. Create and activate a virtual environment
The project was developed and tested with Python 3.12.3
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
All hyperparameters, paths, and experimental settings are defined in a single configuration file: **config.yaml**. This includes:

-environment name

-training hyperparameters

-evaluation settings

-checkpoint paths

-SARFA parameters

### 5. Train an agent
Training is handled via **main.py**.
Choose one algorithm at a time.
#### DDQN
```bash
python3 main.py --train --ddqn
```
#### PPO
```bash
python3 main.py --train --ppo
```
#### SAC
```bash
python3 main.py --train --sac
```
Training outputs and checkpoints are saved under:  runs/<algorithm>/

### 6. Evaluate a trained agent
To evaluate a trained model over multiple episodes:
```bash
python3 main.py --eval --ddqn
python3 main.py --eval --ppo
python3 main.py --eval --sac
```
Choose one algorithm at a time. Evaluation reports mean and standard deviation of episodic returns.

### 7. Run SARFA explanations
SARFA visual explanations can be generated for each trained agent: 
```bash
python3 sarfa.py --algo ddqn
python3 sarfa.py --algo ppo
python3 sarfa.py --algo sac
```
Choose one algorithm at a time. Generated heatmaps are saved under: runs/sarfa/<date>  


### Hardware notes
-GPU acceleration (CUDA) is supported but not required

-The project runs on Linux and WSL

-Training on CPU is significantly slower

### Project structure
**main.py**: training and evaluation entry point

**sarfa.py**: SARFA-based visual explanations

**dqn.py, ppo.py, sac.py**: RL algorithm implementations

**config.yaml**: global configuration file

## References 
- [1] [Explain Your Move: Understanding Agent Actions Using Specific and Relevant Feature Attribution ](https://arxiv.org/abs/1912.12191)
- [2] [ Free invaders online game](https://freeinvaders.org/)
- [3]  [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290)
- [4] [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905)
- [5] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [6] [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)