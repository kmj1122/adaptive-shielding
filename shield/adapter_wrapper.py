"""Shielded OnPolicy Adapter for OmniSafe."""

from __future__ import annotations

from typing import Tuple, Dict

import torch
from rich.progress import track

from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.common.normalizer import Normalizer
from omnisafe.utils.config import Config

from shield.model.constraint_actor_q_and_v_critic import ConstraintActorQAndVCritic
from shield.vectorized_shield import VectorizedShield
import numpy as np
import time


class ShieldedOnPolicyAdapter(OnPolicyAdapter):
    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize an instance of :class:`ShieldedOnPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self.vector_env_nums = num_envs
        self.env_id = env_id
        self._reset_log()

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorQAndVCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
        shield: VectorizedShield,
        normalizer: Normalizer = None,
    ) -> None:
        self._reset_log()
        obs, info = self.reset()
        robot_slices = self._env.get_slices()['robot']
        action_low = torch.from_numpy(self.action_space.low).to(self._device).float()
        action_high = torch.from_numpy(self.action_space.high).to(self._device).float()
        
        agent_pos, agent_mat = shield._process_agent_information(info)
        robot_original_obs = info['original_obs'][:, robot_slices]
        shield.prepare_dp_input(robot_original_obs, agent_pos, agent_mat, device=self._device)
        episode_step = 0
        self.shield_triggered_count = 0
        shield_trigger = False
        weights = torch.zeros(self.vector_env_nums, shield.n_basis).to(self._device)
        shield.coeffs_for_dynamics_prediction = weights
        shield.normalized_coeffs_for_dynamics_prediction = weights
        shield.robot_slices = robot_slices
        # Circle environment has a fixed sigwall location, so call it once here
        is_circle = ('Circle' in self.env_id)
        shield.is_circle = is_circle
        shield.range_limit = info['sigwalls_loc'][0] if is_circle else None

        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            one_step_after_condition = shield.prev_dp_input is not None
            step_condition = episode_step < shield.example_nbr + 1
            
            agent_pos, agent_mat = shield._process_agent_information(info)
            
            if one_step_after_condition and step_condition:
                shield_trigger = shield.update_weights(episode_step)
                shield_trigger = shield_trigger and logger.current_epoch > shield.warm_up_epochs
                
            obs = shield.add_coefficients_to_obs(obs)
            
            original_robot_obs = info['original_obs'][:, robot_slices]
            shield.prepare_dp_input(original_robot_obs, agent_pos, agent_mat, device=self._device)
            act, value_r, value_c, logp = self._get_shielded_actions(obs, info, agent, shield, normalizer, action_low, action_high, shield_trigger)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)
            
            unsafe_condition = torch.from_numpy(shield._check_presafety_condition(info, enhanced_safety=0.)).to(self._device)
            safety_violation = torch.logical_or(unsafe_condition, cost.to(self._device)).float()
            
            self._log_value(reward=reward, cost=cost, info=info)
            logger.store({'Safety/ShieldViolation': safety_violation.cpu().mean()})
            

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})

            logger.store({'Value/reward': value_r})
            buffer.store(
                obs=obs,
                act=act,    
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            episode_step += 1
            logger.store({'Safety/ShieldTriggeredCount': self.shield_triggered_count / episode_step})
            if epoch_end:
                num_dones = int(terminated.contiguous().sum())
                if self._env.num_envs - num_dones:
                    logger.log(
                        f'\nWarning: trajectory cut off when rollout by epoch\
                            in {self._env.num_envs - num_dones} of {self._env.num_envs} environments.',
                    )

            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    
                    if not done:
                        if epoch_end:
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx].float())
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                info['final_observation'][idx].float(),
                            )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                    
                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0
                        self.shield_triggered_count = 0

                    episode_step = 0
                    shield_trigger = False
                    shield.reset()
                    buffer.finish_path(last_value_r, last_value_c, idx)

    def _get_shielded_actions(
        self,
        obs_tensor_for_policy: torch.Tensor,
        info: Dict,
        agent: ConstraintActorQAndVCritic,
        shield: VectorizedShield,
        normalizer: Normalizer,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        shield_trigger: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enhanced shielded action selection with safety memory for vectorized environments."""
        # Check presafety condition
        unsafe_mask = shield._check_presafety_condition(info, enhanced_safety=0.15)
        unsafe_mask = torch.from_numpy(unsafe_mask).to(self._device)
        dones = torch.zeros(self.vector_env_nums, device=self._device).bool()
        # We save robot's position only because conformal prediction is based on robot's position
        shield.update_robot_actual_history(shield.dp_y)
        shield.step_last_triggered += 1
        if shield.shield_triggered:
            # This way, we make sure the conformal prediction scores are correctly updated, time step matching.
            shield.update_conformality_scores()
            shield._set_conformal_thresholds()
            shield.shield_triggered = False

        # if True:
        if shield.step_last_triggered > shield.idle_condition and shield_trigger and shield.prediction_horizon > 0 and unsafe_mask.any():
            agent_pos, _, buttons, goals, hazards, vases, pillars, push_boxes, gremlins, circle = shield.process_info(info, self.vector_env_nums)
            acts, value_r, value_c, logps = agent.sample(obs_tensor_for_policy, n_samples=shield.sampling_nbr)
            action_clipped = acts.clamp(action_low, action_high)
            
            is_safe, min_indices, safety_measure = shield.sample_safe_actions(
                shield.dp_input,
                agent_pos,
                buttons,
                goals,
                hazards,
                vases,
                pillars,
                push_boxes,
                gremlins,
                circle,
                first_action=action_clipped,
                policy=agent.step,
                device=self._device,
                selection_method='top-k',
                k=max(shield.sampling_nbr // 5, 1),
                normalizer=normalizer,
            )
            
            # Save action_clipped and safety_measure with timestamp
            if False:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = f"shield_data_{timestamp}.npz"
                np.savez(
                    save_path,
                    action_clipped=action_clipped,
                    safety_measure=safety_measure
                )
            dones = is_safe | dones    
            act = acts[min_indices, np.arange(len(min_indices))]
            logp = logps[min_indices, np.arange(len(min_indices))]
            
            self.shield_triggered_count += 1
            shield.shield_triggered = True
            shield.step_last_triggered = 0
        else:
            act, value_r, value_c, logp = agent.step(obs_tensor_for_policy)
            shield.shield_triggered = False

        shield.prev_action = act
        return act, value_r, value_c, logp