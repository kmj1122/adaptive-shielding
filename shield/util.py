import pickle
import yaml
import torch.nn as nn
import numpy as np
import torch
from dataclasses import is_dataclass, fields
from typing import Type, TypeVar
from FunctionEncoder import FunctionEncoder
from FunctionEncoder.Wrapper import TimeSeriesWrapper
from FunctionEncoder.Dataset.TransitionDataset import TransitionDataset
from packaging import version  # comes with pip, setuptools, etc.
import sys

T = TypeVar('T')

def derivative_of(x: np.ndarray, dt: float = 0.02) -> np.ndarray:
    """Calculate time derivatives using central difference method for interior points
    and forward/backward differences for endpoints.
    
    Args:
        x: Array of shape (num_pedestrians, history_length) containing position/velocity data
        dt: Time step size in seconds, defaults to 0.02s (50Hz)
        
    Returns:
        Array of same shape as input containing derivatives
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (num_pedestrians, history_length), got shape {x.shape}")
        
    num_peds, history_len = x.shape
    derivatives = np.zeros_like(x)
    
    if history_len < 2:
        return derivatives
    
    # Handle interior points using central difference
    derivatives[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / (2 * dt)
    
    # Handle endpoints
    # Forward difference for first point
    derivatives[:, 0] = (x[:, 1] - x[:, 0]) / dt
    
    # Backward difference for last point
    derivatives[:, -1] = (x[:, -1] - x[:, -2]) / dt
    
    return derivatives

def dict_to_dataclass(data: dict, dataclass_type: Type[T]) -> T:
    """Convert a dictionary to a dataclass instance.
    
    Args:
        data: Dictionary containing configuration values
        dataclass_type: Type of dataclass to create
        
    Returns:
        Instance of the specified dataclass type
    """
    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type.__name__} is not a dataclass")
    
    # Create a dictionary of field values, handling case-insensitive matching
    field_values = {}
    data_lower = {k.upper(): v for k, v in data.items()}
    
    for field in fields(dataclass_type):
        field_name = field.name
        field_name_lower = field_name.upper()
        
        if field_name_lower in data_lower:
            field_values[field_name] = data_lower[field_name_lower]
            
    return dataclass_type(**field_values)


def compute_min_distance(objects_positions, agent_position):
    """Compute minimum distances between agents and objects using JAX.

    Args:
        objects_positions (jnp.ndarray): Array of object positions with shape 
            (sampling_nbr, num envs, num objects, 2)
        agent_position (jnp.ndarray): Array of agent positions with shape 
            (sampling_nbr, num envs, 2)

    Returns:
        jnp.ndarray: Minimum distances for each agent with shape (num envs, num objects)
    """
    agent_xy = agent_position.unsqueeze(-2)
    objects_xy = objects_positions
    distances = torch.norm(objects_xy - agent_xy, dim=-1)
    return torch.min(distances, dim=-1).values

# Returns the desired activation function by name
def get_activation(activation):
    if activation == "relu":
        return nn.relu
    if activation == "relu6":
        return nn.relu6
    elif activation == "tanh":
        return nn.tanh
    elif activation == "sigmoid":
        return nn.sigmoid
    else:
        raise ValueError(f"Unknown activation: {activation}")

def load_data(env_name, data_purpose):
    with open(f'saved_files/env_transitions/{env_name}_{data_purpose}_transitions.pkl', 'rb') as f:
        env_transitions = pickle.load(f)

    return env_transitions

def save_config(config, path):
    yaml_config = {}
    for key, value in config.items():
        if isinstance(value, tuple):
            yaml_config[key] = list(value)
        else:
            yaml_config[key] = value

    with open(path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, indent=2)

def load_config(path):
    with open(path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    restored_config = {}
    tuple_keys = {'input_size', 'output_size'}
    for key, value in yaml_config.items():
        if key in tuple_keys and isinstance(value, list):
            restored_config[key] = tuple(value)
        else:
            restored_config[key] = value
    return restored_config

def load_model(folder_path, task_type, device=None):
    # Determine device with robust CUDA availability check
    if device is None:
        try:
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                # Additional check to ensure CUDA can actually be initialized
                torch.cuda.device_count()
                device = 'cuda'
            else:
                device = 'cpu'
        except Exception as e:
            print(f"WARNING: CUDA check failed with error: {e}. Falling back to CPU.")
            device = 'cpu'
    else:
        # Validate provided device
        if device == 'cuda':
            try:
                if not torch.cuda.is_available():
                    print(f"WARNING: CUDA requested but not available. Falling back to CPU.")
                    device = 'cpu'
                else:
                    torch.cuda.device_count()  # Test CUDA initialization
            except Exception as e:
                print(f"WARNING: CUDA initialization failed with error: {e}. Falling back to CPU.")
                device = 'cpu'
    assert task_type in ['dynamics_predictor', 'mo_predictor'], f"Unknown task type: {task_type}"
    # env_info = folder_path.split('/')[-1].split('-')[0]
    # train_transitions = load_data(env_info, 'train')
    # eval_transitions = load_data(env_info, 'eval')
    # n_functions = len(list(train_transitions['X'].keys()))
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # dataset = TransitionDataset(train_transitions, eval_transitions, n_functions=n_functions, n_examples=100, n_queries=100, dtype=torch.float32, device=device)
    # example_xs, example_ys, query_xs, query_ys, info = dataset.sample(phase='eval')

    config = load_config(f"{folder_path}/config.yaml")
    # Ensure device parameter is set correctly
    config['device'] = device
    print(f"DEBUG: Creating FunctionEncoder with device={device}")
    print(f"DEBUG: torch.cuda.is_available()={torch.cuda.is_available()}")
    model = FunctionEncoder(**config)
    print(f"DEBUG: FunctionEncoder created, moving to device={device}")
    model = model.to(device)
    model.load(f"{folder_path}/model.pth", device=device)
    # coefs, _ = model.compute_representation(example_xs, example_ys, method='least_squares')
    # model.save_initial_coefficients(coefs)

    if task_type == 'dynamics_predictor':
        return model
        
    elif task_type == 'mo_predictor':
        history_length = config['model_kwargs']['encoder_kwargs']['history_length']
        feature_dim = config['model_kwargs']['encoder_kwargs']['input_size']
        model = TimeSeriesWrapper(model, history_length=history_length, feature_dim=feature_dim)
    return model
