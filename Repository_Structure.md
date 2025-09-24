

## Repository Structure
This repository includes the following key directories and files to support our safe reinforcement learning (RL) framework with function encoder representation learning:

- **OmniSafe**: We include the `omnisafe` directory to integrate function encoder representation learning with safe RL algorithms implemented by OmniSafe.
- **FunctionEncoder**: This directory contains the `FunctionEncoder` module, including the transition dataset dataclass and utilities for model saving and loading.
- **Shield**: The `shield` directory houses all supported shielded algorithms, including the shielding mechanism and Safe Reinforced Optimization (SRO).
- **Onpolicy_wrapper.py and adapter_wrapper.py**: These files provide the main components for interacting with safe RL algorithms, facilitating policy training and evaluation.
- **Configuration Files**:
  - Baseline algorithm parameters are located in `omnisafe/configs/on-policy`.
  - Shielding algorithm parameters are located in `omnisafe/configs/shield`.

