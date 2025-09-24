from typing import Dict, List, Tuple, Any, DefaultDict, Optional
import os
from collections import defaultdict
import pickle
import numpy as np
from numpy.typing import NDArray
from tqdm import trange
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.running_mean_std import RunningMeanStd
from copy import deepcopy

def collect_safety_gym_transitions(
    env: Env,
    policy: Optional[PPO] = None,
    num_episodes: int = 100,
    position_only_prediction: bool = False,
) -> DefaultDict[Tuple[float, ...], List[Tuple[NDArray[np.float64], NDArray[np.float64]]]]:
    """Collect transition data from safety gym environment.

    Args:
        env: The gymnasium environment instance
        num_episodes: Number of episodes to collect data from

    Returns:
        Dictionary mapping hidden parameters to lists of (state-action, position_delta) tuples

    Raises:
        ValueError: If environment doesn't provide required information
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    
    obs, info = env.reset()
    slices = env.unwrapped.get_slices()
    transitions_X: DefaultDict[Tuple[float, ...], List[Tuple[NDArray[np.float64], NDArray[np.float64]]]] = defaultdict(list)
    transitions_Y: DefaultDict[Tuple[float, ...], List[Tuple[NDArray[np.float64], NDArray[np.float64]]]] = defaultdict(list)
    transitions = dict()

    obs_rms = RunningMeanStd(shape=env.observation_space.shape)  # type: ignore[assignment, arg-type]
    def normalize_obs(obs: np.ndarray) -> np.ndarray:
        obs_ = deepcopy(obs)
        obs_ = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
        return obs_

    
    for episode in trange(num_episodes, desc="Collecting safety gym transitions"):
        obs, info = env.reset()

        hidden_parameter = tuple(info["hidden_parameters_features"])
        robot_obs = obs[slices['robot']]
        robot_pos = info["agent_pos"][:2]
        robot_mat = info["agent_mat"]


        
        obs_mat_pos_concat = np.concatenate([robot_obs, robot_mat, robot_pos])
        obs_pos_concat = robot_obs

        done = False
        while not done:
            prev_robot_obs = robot_obs.copy()
            prev_robot_pos = robot_pos.copy()
            prev_robot_mat = robot_mat.copy()
            prev_obs_mat_pos_concat = obs_mat_pos_concat.copy()

            if policy is None:
                action = env.action_space.sample()
            else:
                # Order matters
                obs_rms.update(obs)
                obs = normalize_obs(obs)
                action, _ = policy.predict(obs)
                
            obs, _, _, terminated, truncated, info = env.step(action)
                            
            done = truncated | terminated
            robot_obs = obs[slices['robot']]
            robot_pos = info["agent_pos"][:2]
            robot_mat = info["agent_mat"]
            obs_mat_pos_concat = np.concatenate([robot_obs, robot_mat, robot_pos])
            if position_only_prediction:
                x = np.concatenate([prev_robot_obs, action])
                y = robot_pos - prev_robot_pos
            else:
                x = np.concatenate([prev_obs_mat_pos_concat, action])
                pos_delta = robot_pos - prev_robot_pos
                mat_delta = robot_mat - prev_robot_mat
                obs_delta = robot_obs - prev_robot_obs
                y = np.concatenate([obs_delta, pos_delta, mat_delta])
            transitions_X[hidden_parameter].append(x)
            transitions_Y[hidden_parameter].append(y)

    transitions = {
        'X': transitions_X,
        'Y': transitions_Y
    }
    env.close()
    return transitions

def save_transitions(
    train_transitions: Dict[Any, List[Any]],
    eval_transitions: Dict[Any, List[Any]],
    env_id: str,
    default_path: str = "."
) -> None:
    """Save collected transitions to pickle files.

    Args:
        train_transitions: Training data transitions
        eval_transitions: Evaluation data transitions
        env_id: Environment identifier string
        default_path: Directory path to save files

    Raises:
        ValueError: If env_id is invalid
        OSError: If directory creation or file writing fails
    """
    if not env_id or not isinstance(env_id, str):
        raise ValueError("Invalid env_id provided")

    train_filename = f"{env_id.split('-')[0][:-1]}_train_transitions.pkl"
    eval_filename = f"{env_id.split('-')[0][:-1]}_eval_transitions.pkl"
    os.makedirs(default_path, exist_ok=True)

    train_path = os.path.join(default_path, train_filename)
    eval_path = os.path.join(default_path, eval_filename)

    with open(train_path, "wb") as f:
        pickle.dump(train_transitions, f)
    print(f"Training transitions saved to {train_filename}")

    with open(eval_path, "wb") as f:
        pickle.dump(eval_transitions, f)
    print(f"Evaluation transitions saved to {eval_filename}")