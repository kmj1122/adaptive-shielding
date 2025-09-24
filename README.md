# Runtime Safety through Adaptive Shielding: From Hidden Parameter Inference to Provable Guarantees

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Table of Contents
- [Overview](#-overview)
- [Installation](#-installation)
- [RepositoryStructure](#-repositorystructure)
- [Usage](#-usage)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)


## RepositoryStructure
This repository includes the following key directories and files to support our safe reinforcement learning (RL) framework with function encoder representation learning:

- **OmniSafe**: We include the `omnisafe` directory to integrate function encoder representation learning with safe RL algorithms implemented by OmniSafe.
- **FunctionEncoder**: This directory contains the `FunctionEncoder` module, including the transition dataset dataclass and utilities for model saving and loading.
- **Shield**: The `shield` directory houses all supported shielded algorithms, including the shielding mechanism and Safe Reinforced Optimization (SRO).
- **Onpolicy_wrapper.py and adapter_wrapper.py**: These files provide the main components for interacting with safe RL algorithms, facilitating policy training and evaluation.
- **Configuration Files**:
  - Baseline algorithm parameters are located in `omnisafe/configs/on-policy`.
  - Shielding algorithm parameters are located in `omnisafe/configs/shield`.


## Overview
**Adaptive Shielding** is a framework that combines safety-related objectives (SRO) with learned dynamics models to actively shield RL agents from unsafe actions in environments with hidden dynamics. Our approach:

- Builds on **Constrained Hidden-Parameter MDPs**
- Uses **Function Encoders** for real-time inference of unobserved parameters
- Employs **conformal prediction** to provide probabilistic safety guarantees with minimal runtime overhead

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Download texture assets:**
   ```bash
   # For users with the zip file distribution
   # Download textures from https://github.com/PKU-Alignment/safety-gymnasium/tree/main/safety_gymnasium/assets/textures
   # Then, place in the correct directory
   mv textures envs/safety_gymnasium/assets/
   ```

## Usage

Follow these steps in order to train, shield, and evaluate your RL agent:

### 1. (Optional) Pre-train Policies for Data Collection

```bash
# Arguments: <env_id> <timesteps>
python 0.train_policies.py SafetyPointGoal1-v1 2000000
```

### 2. Collect Transitions Dataset

```bash
# Arguments: <env_id> <num_episodes> <use_trained_policy>
# Set use_trained_policy=1 if you completed step 1, otherwise 0 for random policy
python 1.collect_transition.py SafetyPointGoal1-v1 1000 0
```

### 3. Train Function Encoder

Function encoder settings can be configured in the `configuration.py` file.

```bash
# Arguments: <env_id> <seed> <use_wandb>
python 2.compare_dynamic_predictors.py --env_id SafetyPointGoal1-v1
```

### 4. Train with Adaptive Shielding

```bash
# Generic command
python run.py \
  --env_id <env_id> \
  --algo <algorithm> \
  --prediction_horizon <0|1> \
  --penalty-type <reward|shield> \
  --sampling-nbr <sampling_number> \
  --safety-bonus <bonus_weight> \
  --idle-condition 4 \
  --use-wandb <True|False> \
  --fe-representation <True|oracle> \
  --project-name <project_name>
```

**Available algorithms:**
- Shielded algorithms: `ShieldedTRPOLag`, `ShieldedPPOLag`, `ShieldedRCPO`
- Baseline algorithms: `PPOLag`, `TRPOLag`, `CUP`, `CPO`, `TRPOSaute`, `PPOSaute`, `FOCOPS`, `RCPO`, `RCPOSaute` (these use oracle representation automatically, unless it's specified)

**Example command:**
```bash
python run.py \
  --env-id SafetyPointGoal1-v1 \
  --algo ShieldedTRPOLag \
  --prediction-horizon 1 \
  --penalty-type reward \
  --sampling-nbr 10 \
  --safety-bonus 1. \
  --idle-condition 4 \
  --use-wandb True \
  --fe-representation True \
  --project-name shield 
```

#### Key Parameters:
- `--prediction-horizon`: 0 (no shielding), 1 (one-step ahead shielding)
- `--penalty-type`: `reward` (use SRO), `shield` (do not use SRO during optimization)
- `--fe-representation`: `True` (function encoder adaptation), `oracle` (ground-truth adaptation)
- `--sampling-nbr`: Number of action samples when adaptive shield is triggered
- `--safety-bonus`: Weight of safety in the augmented objective
- `--idle-condition`: Control frequent Shielding trigger, letting terms between activation of the Shielding

#### Common Configurations:

| Mode                     | Parameters                                    |
|--------------------------|-----------------------------------------------|
| **SRO only**             | `--prediction-horizon 0 --penalty-type reward` |
| **Shield only**          | `--prediction-horizon 1 --penalty-type shield` |
| **Combined approach**    | `--prediction-horizon 1 --penalty-type reward` |

### 5. Run Unrolling Safety Layer (USL) Baseline

```bash
# Generic command
python run_usl.py --env <env_id> --use_usl --seed <seed> --oracle --save_model

# Example command
python run_usl.py --env SafetyPointGoal1-v1 --use_usl --seed 0 --oracle --save_model
```

### 6. Evaluate OOD Generalization
After training, you can find the trained model on generated `runs` folders. 
You have to specify the trained model's epoch to specific folders. In my case, I used the pattern 
`model_path = f"./results/fe/{env_info[:-1]}1-v1/{algorithm}/seed{seed}/torch_save/epoch-100.pt"`

For OOD testing, use environments with level 2 (e.g., SafetyPointGoal2-v1). These environments have:
- 2 additional hazard spaces 
- Hidden parameters sampled from OOD range
- Use `prediction_horizon=1` to enable shielding, `0` to disable it

```bash
# Generic command
python 3.load_model.py <env_id> <algorithm> <seed> <sampling_nbr> <prediction_horizon>

# Example command
python 3.load_model.py SafetyPointGoal2-v1 ShieldedTRPO 0 100 1
```


## Acknowledgements

This code leverages and extends:
- [saferl_kit](https://github.com/zlr20/saferl_kit)
- [OmniSafe](https://github.com/PKU-Alignment/omnisafe)
- [Safe-Gym](https://github.com/PKU-Alignment/safety-gymnasium)
- [FunctionEncoder](https://github.com/tyler-ingebrand/FunctionEncoder)

## License

Distributed under the MIT License. See `LICENSE` for details. 