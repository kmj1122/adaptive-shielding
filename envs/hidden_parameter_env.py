from typing import Any, ClassVar, Union, Tuple, List, Dict, Optional
import torch
import random
from gymnasium import spaces
import numpy as np
from omnisafe.envs.core import CMDP, env_register, env_unregister
import envs
from envs import make
from envs.safety_gymnasium.vector import SafetySyncVectorEnv
from envs.safety_gymnasium.configuration import EnvironmentConfig
from shield.util import dict_to_dataclass


class HiddenParameterEnv(CMDP):
    _support_envs: ClassVar[List[str]] = ['SafetyPointGoal1-v1', 'SafetyPointGoal2-v1',
                                          'SafetyCarGoal1-v1', 'SafetyCarGoal2-v1',
                                          'SafetyPointButton1-v1', 'SafetyPointButton2-v1',
                                          'SafetyCarButton1-v1', 'SafetyCarButton2-v1',
                                          'SafetyPointCircle1-v1', 'SafetyPointCircle2-v1',
                                          'SafetyCarCircle1-v1', 'SafetyCarCircle2-v1',
                                          'SafetyCarPush1-v1', 'SafetyCarPush2-v1',
                                          'SafetyPointPush1-v1', 'SafetyPointPush2-v1',
                                          'SafetyPointCustomGoal1-v1', 'SafetyPointCustomGoal2-v1',
                                          'SafetyPointCustomButton1-v1', 'SafetyPointCustomButton2-v1',
                                          ]

    need_auto_reset_wrapper = False  # Whether `AutoReset` Wrapper is needed
    need_time_limit_wrapper = False  # Whether `TimeLimit` Wrapper is needed

    def __init__(self, env_id: str, device: str, env_config: EnvironmentConfig, num_envs: int = 1, render_mode: Optional[str] = None, camera_id: Optional[int] = None, **kwargs) -> None:
        if isinstance(env_config, dict):
            env_config = dict_to_dataclass(env_config, EnvironmentConfig)
        self.render_mode = render_mode
        self.oracle = env_config.USE_ORACLE
        self.n_basis = env_config.NBR_OF_BASIS
        self._num_envs = num_envs
        env_fns = [lambda: make(env_id[:-1] + str(0), env_config=env_config, render_mode=render_mode, camera_id=camera_id) for i in range(num_envs)]
        self.env = SafetySyncVectorEnv(env_fns)
        
        _, self._initial_info = self.env.reset()
        # For oracle, we the true hidden parameters, otherwise we use the n_basis of function encoder, which is a proxy for the hidden parameters
        self.hidden_parameters_dim = self._initial_info['hidden_parameters_dim'][0] if self.oracle else self.n_basis
        
        self._device = torch.device(device)
        self._count = 0
        self._num_envs = num_envs
        self.obs_dims = {k: v[0] for k, v in self._initial_info['obs_dims'].items() if not k.startswith('_')}
        self.slices = self.get_slices()

        self._org_observation_space = self.env.single_observation_space
        
        bounds = {'original': (self._org_observation_space.low, self._org_observation_space.high)}   
        bounds['hidden'] = ([-2.] * self.hidden_parameters_dim, [2.] * self.hidden_parameters_dim)

        low = np.concatenate([b[0] for b in bounds.values()])
        high = np.concatenate([b[1] for b in bounds.values()])

        self._observation_space = spaces.Box(low, high)
        self._action_space = self.env.single_action_space
        
    def get_slices(self) -> dict:
        slices = {}
        dim_idx = 0
        for obs_key, obs_dim in self.obs_dims.items():
            if obs_key == 'hidden_parameters_features':
                continue
            slices[obs_key] = slice(dim_idx, dim_idx + obs_dim)
            dim_idx += obs_dim
        return slices

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

    def reset(
        self,
        seed: Union[int, None] = None,
        options: Union[Dict[str, Any], None] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if seed is not None:
            self.set_seed(seed)

        obs, info = self.env.reset(seed=seed)
        obs = self.augment_state(obs, info)
        obs = torch.as_tensor(obs, device=self._device).float()
        return obs, info
    
    def augment_state(self, obs: np.ndarray, info: dict) -> np.ndarray:
        hidden_info = np.vstack(info['hidden_parameters_features']).reshape(self._num_envs, -1) if self.oracle else np.zeros((self._num_envs, self.hidden_parameters_dim))
        return np.concatenate((obs, hidden_info), axis=1)

    @property
    def max_episode_steps(self) -> None:
        """The max steps per episode."""
        if 'Circle' in self.env.spec.id:
            return 500
        else:
            return 1000

    def render(self) -> Any:
        return self.env.render()
    
    def render_rgb_array(self) -> np.ndarray:
        return self.env.render()

    def close(self) -> None:
        self.env.close()
    
    def step(
        self,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, reward, cost, terminated, truncated, info = self.env.step(action.cpu().detach().numpy())
        obs = self.augment_state(obs, info)
        obs = torch.as_tensor(obs, device=self._device, dtype=torch.float32)
        reward = torch.as_tensor(reward, device=self._device, dtype=torch.float32)
        cost = torch.as_tensor(cost, device=self._device, dtype=torch.float32)
        terminated = torch.as_tensor(terminated, device=self._device, dtype=torch.float32)
        truncated = torch.as_tensor(truncated, device=self._device, dtype=torch.float32)
        # Handle final observation
        if torch.any(terminated) or torch.any(truncated):
            # Convert to boolean before OR operation
            info['_final_observation'] = (terminated > 0) | (truncated > 0)
            info['final_observation'] = torch.as_tensor(obs, device=self._device, dtype=torch.float32)
        return obs, reward, cost, terminated, truncated, info

@env_register
class HiddenParamEnvs(HiddenParameterEnv):
    example_configs = 2