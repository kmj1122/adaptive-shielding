
import os
import pickle

import sys
import torch
from stable_baselines3 import PPO
import numpy as np
from envs.safety_gymnasium.configuration import EnvironmentConfig
from shield.dataset import collect_safety_gym_transitions
from envs import make
import envs

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def save_data(train_transitions, env_info, data_purpose: str = 'train', trainsition_save: bool = False):
    if trainsition_save:
        os.makedirs('saved_files/env_transitions', exist_ok=True)
        with open(f'saved_files/env_transitions/{env_info}_{data_purpose}_transitions.pkl', 'wb') as f:
            pickle.dump(train_transitions, f)

def load_data(env_name, data_purpose):
    with open(f'saved_files/env_transitions/{env_name}_{data_purpose}_transitions.pkl', 'rb') as f:
        env_transitions = pickle.load(f)

    return env_transitions

# python 1.collect_transition.py env_id=SafetyPointGoal1-v0 nbr_of_episodes=10 use_trained_policy=0
def collect_transition(env_id: str, nbr_of_episodes: int, use_trained_policy: bool):
    env_config = EnvironmentConfig()
    env_info = env_id.split('-')[0]
    use_trained_policy = 0
    env_config.FIX_HIDDEN_PARAMETERS = False

    # If we want to use the trained policy to collect transition dynamics, we need to load the trained policy
    if use_trained_policy:
        log_dir = f"./trained_policies_for_collection/{env_id}/"
        model = PPO.load(os.path.join(log_dir, "best_model"))
    # If we don't want to use the trained policy, we can use a random policy
    else:
        model = None

    # We collect transition dynamics for both training and evaluation
    for data_purpose in ['train', 'eval']:
        env = make(env_info + '-v0', env_config=env_config, render_mode='rgb_array')
        if data_purpose == 'eval':
            # For evaluation, we only use 20 perecent of the episodes during training
            nbr_of_episodes = nbr_of_episodes // 5 
        
        transitions = collect_safety_gym_transitions(env, policy=model, num_episodes=nbr_of_episodes, position_only_prediction=True)
        save_data(transitions, env_info, data_purpose=data_purpose, trainsition_save=True)

if __name__ == "__main__":
    env_id = sys.argv[1]
    nbr_of_episodes = int(sys.argv[2])
    use_trained_policy = int(sys.argv[3])
    collect_transition(env_id=env_id, nbr_of_episodes=nbr_of_episodes, use_trained_policy=use_trained_policy)