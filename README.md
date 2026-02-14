# Project for RL exam 2026
<!-- Badges -->
[![Python](https://img.shields.io/badge/python-3.12.3-blue)](https://www.python.org/)
[![Camilla Giuliani on GitHub](https://img.shields.io/badge/Camilla–Giuliani–GitHub-181717?style=plastic&logo=github)](https://github.com/camygiuliani)
[![Pietro D'Annibale on GitHub](https://img.shields.io/badge/Pietro–D%E2%80%99Annibale–GitHub-181717?style=plastic&logo=github)](https://github.com/Sassotek)
<!--........-->

Camilla Giuliani 1883207 &&  Pietro D'Annibale 1917211
<!-- ## Project structure
**config.yaml**: global configuration file 

**main.py**: training and evaluation entry point

**dqn.py, ppo.py, sac.py**: RL algorithm implementations

**sarfa.py**: SARFA-based visual explanations

**robustness.py** code for robustness test and boh???

**utils.py** and **wrappers.py** for other functions
 -->

## Project structure

**Core**
- `main.py` — training & evaluation entry point  
- `ddqn.py`, `ppo.py`, `sac.py` — RL agents
- `compare_eval.py` — comparative results for all the agents  

**Interpretability**
- `sarfa.py` — SARFA visual explanations (heatmaps, comparison grids, videos)

**Robustness**
- `robustness.py` — patch occlusion tests (random baseline vs SARFA-guided)

**Environment & utilities**
- `wrappers.py` — environment creation and preprocessing  
- `utils.py` — common helper functions  
- `config.yaml` — global configuration

**Outputs**
- `runs/` — training checkpoints and logs (timestamped)
- `robustness_outputs/` — robustness results
- `sarfa_outputs/` — SARFA visual outputs (timestamped)


## Experimental Setup (Summary)

- **Benchmark:** Atari Space Invaders (ALE v5)
- **State representation:** 84×84 grayscale frames with **4-frame stacking**
- **Action space:** 6 discrete actions
- **Agents compared:** **DDQN**, **PPO**, **SAC (discrete)**
- **Interpretability:** **SARFA** visual explanations (heatmaps + video overlays)
- **Robustness evaluation:** patch occlusion tests (**random baseline** vs **SARFA-guided top-K occlusion**)



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

### 3a. Additional libraries
Needed to fix  warning alsa sound
```bash
sudo apt install -y alsa-utils libasound2t64
sudo apt install -y pulseaudio
```
Disclaimer: audio may not work even though you install these libraries above.


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

```bash
python3 main.py --train --ddqn 
python3 main.py --train --ppo
python3 main.py --train --sac
```
> Additional hyperparameters and settings can be customized in `config.yaml` and directly in `main.py`.

Training outputs and checkpoints are saved under:  `runs/<algorithm>/` and   `checkpoints/<algorithm>/` and both  and are further organized into timestamped subfolders.

### 6. Evaluate a trained agent
To evaluate a trained model over multiple episodes:
```bash
python3 main.py --eval --ddqn
python3 main.py --eval --ppo
python3 main.py --eval --sac
```
Choose one algorithm at a time. Evaluation reports mean and standard deviation of episodic returns.
> Additional hyperparameters and settings can be customized in `config.yaml` and directly in `main.py`.

Evaluation outputs are saved under:  `runs/<algorithm>/` and are further organized into timestamped subfolders.


### 7a. Run SARFA explanations (snapshots)
SARFA visual explanations can be generated for each trained agent. Choose one algorithm at a time:
```bash
python3 sarfa.py --algo ddqn
python3 sarfa.py --algo ppo
python3 sarfa.py --algo sac
```
 or to  generate a 3×3 comparison grid that  shows a SARFA visual explanation for each algorithm at three different moments of the episode (early, mid, and late) : 
```bash
python3 sarfa.py --algo all
```   
> Additional hyperparameters and settings can be customized in `config.yaml` and directly in `sarfa.py`.

Generated heatmaps are saved under `sarfa_outputs/` and are further organized into timestamped subfolders.


### 7b. Run SARFA explanations (single full-screen video + colored heatmap overlay)
```bash
python3 sarfa.py --algo ddqn --video
python3 sarfa.py --algo ppo --video 
python3 sarfa.py --algo sac --video
```   
> Additional hyperparameters and settings can be customized in `config.yaml` and directly in `sarfa.py`.

The videos are saved under `sarfa_outputs/` and are further organized into timestamped subfolders.


### 8 Robustness analysis


```bash
python3 robustness.py --algo <ddqn|ppo|sac>
```
To enable the SARFA-guided occlusion condition, add:
```bash
python3 robustness.py --algo <ddqn|ppo|sac> --run_sarfa
```
> Additional hyperparameters and settings can be customized in `config.yaml` and directly in `robustness.py`.

Robustness outputs are saved under:  `robustness_outputs/<algorithm>/` and are further organized into timestamped subfolders.




## Hardware notes
-GPU acceleration (CUDA) is supported but not required

-The project runs on Linux and WSL

-Training on CPU is significantly slower

## License
MIT License — see [LICENSE](LICENSE).


## References 
- [1] [Explain Your Move: Understanding Agent Actions Using Specific and Relevant Feature Attribution ](https://arxiv.org/abs/1912.12191)
- [2]  [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290)
- [3] [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905)
- [4] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [5] [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)