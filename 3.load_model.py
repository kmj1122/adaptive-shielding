"""Script to load and evaluate a saved TRPO model from OmniSafe."""
import os
import sys
import numpy as np
import pandas as pd
import torch
from envs.safety_gymnasium.configuration import EnvironmentConfig
from envs.hidden_parameter_env import HiddenParamEnvs
from shield.vectorized_shield import VectorizedShield
from omnisafe.evaluator_with_shield import Evaluator
from load_utils import load_model, load_config, set_seed
from shield.util import load_model as load_model_shield



def main(env_id: str, algorithm: str, seed: int, sampling_nbr: int, prediction_horizon: int, threshold: float, idle_condition: int):
    """Main function to load and evaluate the model."""
    env_info = env_id.split('-')[0]
    save_env_info = env_info
    if 'Shielded' not in algorithm:
        model_path = f"./results/oracle/{env_info[:-1]}1-v1/Oracle-{algorithm}/seed{seed}/torch_save/epoch-100.pt"
        config_path = f"./results/oracle/{env_info[:-1]}1-v1/Oracle-{algorithm}/seed{seed}/config.json"
        load_path = f"./results/oracle/{env_info[:-1]}1-v1/Oracle-{algorithm}/seed{seed}"
    elif 'Shielded' in algorithm:
        model_path = f"./results/fe/{env_info[:-1]}1-v1/{algorithm}/seed{seed}/torch_save/epoch-100.pt"
        config_path = f"./results/fe/{env_info[:-1]}1-v1/{algorithm}/seed{seed}/config.json"
        load_path = f"./results/fe/{env_info[:-1]}1-v1/{algorithm}/seed{seed}"
        
    # to get the same result
    set_seed(0)
    device = 'cpu'
    config = load_config(config_path)
    # render_mode = 'human'
    env_config = EnvironmentConfig()
    env_config.IS_OUT_OF_DISTRIBUTION = True
    env_config.FIX_HIDDEN_PARAMETERS = False
    if 'Shielded' not in algorithm:
        env_config.USE_ORACLE = True
    
    unwrapped_env = HiddenParamEnvs(env_id, device=device, env_config=env_config, num_envs=1, render_mode='rgb_array')    
    
    
    # Load the model and config
    model, env, config = load_model(model_path, config, unwrapped_env, device=device)
    if 'Shielded' in algorithm and 'SRO' not in algorithm:
        dynamic_model = load_model_shield(f'saved_files/dynamics_predictor/{env_info[:-1]}1-v1', 'dynamics_predictor', device=device)
        shield = VectorizedShield(dynamic_predictor=dynamic_model, mo_predictor=None, sampling_nbr=sampling_nbr, prediction_horizon=1, vector_env_nums=1, static_threshold=threshold, example_nbr=100, idle_condition=idle_condition)
        shield.reset()
        evaluator = Evaluator(env=env, unwrapped_env=unwrapped_env, actor=model, shield=shield, safety_budget=config.get('safety_budget', 1.0))
    else:
        shield = None
        evaluator = Evaluator(env=env, unwrapped_env=unwrapped_env, actor=model.actor, shield=None, safety_budget=config.get('safety_budget', 1.0))
    
    evaluator.load_saved(save_dir=load_path, render_mode="rgb_array", env=env)
    if 'Shielded' in algorithm and 'SRO' not in algorithm and 'b' in algorithm:
        output_dir = f"./ood_evaluation_folder/{save_env_info}/{algorithm}_{sampling_nbr}_{threshold}_{idle_condition}/seed{seed}"
    if 'Shielded' in algorithm and 'SRO' not in algorithm and 'b' not in algorithm:
        output_dir = f"./ood_evaluation_folder/{save_env_info}/Shield_{sampling_nbr}_{threshold}_{idle_condition}/seed{seed}"
    elif 'Shielded' in algorithm and 'SRO' in algorithm:
        output_dir = f"./ood_evaluation_folder/{save_env_info}/SRO/seed{seed}"
    else:
        output_dir = f"./ood_evaluation_folder/{save_env_info}/{algorithm}/seed{seed}"
    output_file_path = os.path.join(output_dir, "evaluation_results.csv")
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}")
    episode_rewards, episode_costs, episode_lengths, shield_trigger_counts, episode_run_times, episode_hidden_parameters = evaluator.evaluate(num_episodes=100, cost_criteria=1.0, save_plot=False, seed=seed)
    
    # Save episode rewards and hidden parameters using numpy's .npz format
    data_file_path = os.path.join(output_dir, "episode_data.npz")
    np.savez(
        data_file_path,
        rewards=np.array(episode_rewards),
        hidden_parameters=np.array(episode_hidden_parameters),
        costs=np.array(episode_costs),
        lengths=np.array(episode_lengths),
        shield_trigger_counts=np.array(shield_trigger_counts),
        run_times=np.array(episode_run_times)
    )
        
    # Prepare data for DataFrame
    results_data = {
        "Metric": [
            "Average episode reward",
            "Average episode cost",
            "Average episode length",
            "Average shield triggered",
            "Average episode run time"
        ],
        "Value": [
            np.mean(a=episode_rewards),
            np.mean(a=episode_costs),
            np.mean(a=episode_lengths),
            np.mean(a=shield_trigger_counts),
            np.mean(a=episode_run_times)
        ]
    }
    results_df = pd.DataFrame(results_data)

    # Save the DataFrame to CSV
    results_df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    # env_id = "SafetyPointGoal2-v1"  # You can change this to match your environment
    # python 4.load_model.py SafetyCarCircle2-v1 ShieldedRCPO_s50_b1.0_i4 0 5 1 0.2 4
    env_id = sys.argv[1]
    algorithm = sys.argv[2]
    seed = int(sys.argv[3])
    sampling_nbr = int(sys.argv[4])
    prediction_horizon = int(sys.argv[5])
    threshold = float(sys.argv[6])
    idle_condition = int(sys.argv[7])
    main(env_id, algorithm, seed, sampling_nbr, prediction_horizon, threshold, idle_condition) 