from typing import Dict, Tuple
import torch
import numpy as np
from shield.base_shield import BaseShield
from shield.util import compute_min_distance
from omnisafe.envs.wrapper import Normalizer
from FunctionEncoder import FunctionEncoder


class VectorizedShield(BaseShield):
    """Shield structure for RL algorithms that produces safer policies.
    
    Takes a policy and observations to generate safer actions by predicting future states
    and checking for potential collisions.
    
    Attributes:
        scene: Scene information for efficient use of obstacle history
        dynamic_predictor: Predicts agent's future states
        moving_obstacles_predictor: Predicts moving obstacles' future states  
        prediction_horizon: Number of prediction steps
    """

    def __init__(
        self,
        dynamic_predictor: FunctionEncoder,
        mo_predictor: FunctionEncoder,
        sampling_nbr: int,
        prediction_horizon: int,
        safety_bonus: float = 1.0,
        window_size: int = 100,
        significance_level: float = 0.1,
        static_threshold: float = 0.225,
        mo_threshold: float = 0.225,
        example_nbr: int = 100,
        warm_up_epochs: int = 0,
        idle_condition: int = 4,
        **kwargs,
    ) -> None:
        self.is_object_checked = False
        self.static_threshold = static_threshold
        self.mo_threshold = mo_threshold
        self.example_nbr = example_nbr
        self.safety_bonus = safety_bonus
        self.shield_triggered = False
        self.warm_up_epochs = warm_up_epochs
        """Initialize the shield.

        Args:
            env_id: Environment identifier
            dynamic_predictor_cfgs: Configuration for dynamic state predictor
            moving_obstacles_predictor_cfgs: Configuration for obstacle predictor
            sampling_nbr: Number of samples for prediction
            prediction_horizon: Number of steps to predict ahead
            threshold: Safety threshold distance
            discount_factor: Discount factor for future predictions
            use_hidden_param: Whether to use hidden parameters
            use_online_update: Whether to update online
            window_size: Window size for predictions
            significance_level: Statistical significance level
            safety_bonus: Safety bonus
            gradient_scale: Gradient scale
            warm_up_epochs: Number of warm-up epochs
        """
        super().__init__(
            dynamic_predictor=dynamic_predictor,
            mo_predictor=mo_predictor,
            sampling_nbr=sampling_nbr,
            prediction_horizon=prediction_horizon,
            window_size=window_size,
            significance_level=significance_level,
            idle_condition=idle_condition,
            **kwargs,
        )
        self.xs_history = []
        self.ys_history = []
        
    def update_weights(self, step):
        shield_trigger = False
        example_x = torch.cat([self.prev_dp_input, self.prev_action], axis=-1).detach()
        example_y = self.dp_y
        
        self.xs_history.append(example_x.unsqueeze(1))
        self.ys_history.append(example_y.unsqueeze(1))
        
        example_xs = torch.cat(self.xs_history, dim=1)
        example_ys = torch.cat(self.ys_history, dim=1)

        if step == self.example_nbr:
            weights = self._compute_coefs(example_xs, example_ys)
            # We normalize the weights by the number of basis functions square for stability
            norm = torch.norm(weights, dim=1).reshape(-1, 1) * self.n_basis ** 4
            self.normalized_coeffs_for_dynamics_prediction = weights / norm
            self.xs_history = []
            self.ys_history = []
            shield_trigger = True
            
        return shield_trigger
        
    def _process_agent_information(self, info: Dict):
        """Process and normalize agent position from environment info."""
        agent_pos = np.stack(info['agent_pos'])
        agent_mat = np.stack(info['agent_mat'])
        if len(agent_pos.shape) == 1:
            agent_pos = agent_pos.reshape(1, -1)
        if len(agent_mat.shape) == 1:
            agent_mat = agent_mat.reshape(1, -1)
        
        return agent_pos, agent_mat

    def _check_presafety_condition(self, info: Dict, enhanced_safety: float = 0.0):
        """Check if the current state satisfies safety conditions."""
        if self.is_circle:
            agent_pos = np.array([pos[:2] for pos in info["agent_pos"]])
            range_limit_check = np.abs(agent_pos) > self.range_limit - 0.125
            unsafe_condition = np.any(range_limit_check, axis=1, keepdims=False)
        else:
            unsafe_condition = info['min_distance'] < self.static_threshold + enhanced_safety
            
        return unsafe_condition

    @torch.no_grad()
    def _compute_coefs(self, example_xs, example_ys):
        coefs, _ = self.dynamic_predictor.compute_representation(example_xs, example_ys, method='least_squares')
        self.coeffs_for_dynamics_prediction = coefs
        return coefs
        
    def process_info(self, info: Dict, batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process environment info to extract relevant elements.

        Args:
            info: Dictionary containing environment information

        Returns:
            Tuple containing:
            - Agent position and orientation matrix
            - Goal position
            - Button positions
            - Static obstacle positions
            - Moving obstacle positions
            - Moving obstacle z-coordinates
            - Hidden parameters
        """    
        
        agent_pos = (
            np.array([pos for pos in info["agent_pos"]]) 
            if "agent_pos" in info else np.zeros((batch_size, 3))
        )

        agent_mat = (
            np.array([mat for mat in info["agent_mat"]])
            if "agent_mat" in info else np.zeros((batch_size, 9))
        )

        buttons = (
            np.array([button for button in info["buttons"]])
            if "buttons" in info else np.array([])
        )

        goal_pos = (
            np.array([pos for pos in info["goal_pos"]])
            if "goal_pos" in info else np.array([])
        )

        hazards = (
            np.array([hazard for hazard in info["hazards"]])
            if "hazards" in info else np.array([])
        )
        vases = (
            np.array([vase for vase in info["vases"]])
            if "vases" in info else np.array([])
        )
        pillars = (
            np.array([pillar for pillar in info["pillars"]])
            if "pillars" in info else np.array([])
        )
        push_boxes = (
            np.array([push_box for push_box in info["push_box"]])
            if "push_box" in info else np.array([])
        )
        gremlins = (
            np.array([gremlin for gremlin in info["gremlins"]])
            if "gremlins" in info else np.array([])
        )
        circle = (
            np.array([circle for circle in info["circle"]])
            if "circle" in info else np.array([])
        )
        
        return agent_pos, agent_mat, buttons, goal_pos, hazards, vases, pillars, push_boxes, gremlins, circle

    def sample_safe_actions(
        self,
        dp_input: np.ndarray,
        agent_pos: np.ndarray,
        buttons: np.ndarray,
        goal_pos: np.ndarray,
        hazards: np.ndarray,
        vases: np.ndarray,
        pillars: np.ndarray,
        push_boxes: np.ndarray,
        gremlins: np.ndarray,
        circle: np.ndarray,
        first_action: np.ndarray,
        policy, 
        device: str = 'cpu',
        selection_method: str = 'top-k',
        k: int = 10,
        normalizer: Normalizer = None,
        early_return: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # check if the object is present in the environment
        if not self.is_object_checked:
            self.is_gremlins = gremlins.shape[0] > 0
            self.is_push_boxes = push_boxes.shape[0] > 0
            self.is_vases = vases.shape[0] > 0
            self.is_hazards = hazards.shape[0] > 0
            self.is_pillars = pillars.shape[0] > 0
            self.is_buttons = buttons.shape[0] > 0
            self.is_goal = goal_pos.shape[0] > 0
            self.is_circle = circle.shape[0] > 0
            self.is_object_checked = True            

        # Repeat dp_input to match first_action's first dimension
        # dp_input shape: (4, 12) -> (sampling_nbr, vector_num_env, robot_dim + action_dim)
        self.shield_triggered = True
        dp_input_repeated = dp_input.unsqueeze(0).repeat(self.sampling_nbr, 1, 1)

        obs_dp_input = (
            torch.cat([dp_input_repeated, first_action], dim=-1)  # → (sampling_nbr, num_env, robot+action)
            .transpose(0, 1)                                # → (num_env, sampling_nbr, robot+action)
            .contiguous()                                   # (optional) for safe downstream reshaping
        )
        
        with torch.no_grad():
            robot_delta_predictions = self.dynamic_predictor.predict(obs_dp_input, self.coeffs_for_dynamics_prediction).transpose(0, 1) # (sampling_nbr, num_env, output_size)

        # This is for multi-step prediction 
        # robot_predictions = (dp_input_repeated + robot_delta_predictions).detach()
        # robot_xy_predictions = robot_predictions[:, :, -12:-10]
        # robot_xyz_predictions = robot_predictions[:, :, -12: -9]
        # robot_matrix_predictions = robot_predictions[:, :, -9:]
        # prediction_step = 0
        sampling_nbr, num_envs, _ = robot_delta_predictions.shape
        robot_xy_predictions = robot_delta_predictions + torch.from_numpy(agent_pos[np.newaxis, :, :2]).float().to(device)

        # mean_robot_predictions = torch.mean(robot_predictions, axis=0)
        # self.robot_predictions_history.append(mean_robot_predictions.detach())        
        mean_robot_xy_predictions = torch.mean(robot_xy_predictions, axis=0)
        self.robot_predictions_history.append(mean_robot_xy_predictions.detach())        
        
        if self.is_gremlins and self.mo_last_obs is not None:
            predicted_gremlins, _ = self.get_mo_predictions()
            for i in range(self.prediction_horizon):
                self.mo_predictions_history.append(predicted_gremlins[:, :, i, :])

            vectorized_gremlins = predicted_gremlins.reshape(num_envs, -1, self.prediction_horizon, 2).unsqueeze(0).repeat(self.sampling_nbr, 1, 1, 1, 1)        

        conformal_threshold = min(self.robot_conformal_threshold, 0.1)
        static_adjusted_threshold = self.static_threshold + conformal_threshold
        mo_adjusted_threshold = self.mo_threshold + conformal_threshold
        range_limit_adjusted_threshold = 0.1 + conformal_threshold
        
        # weighted_distance_min will be safety measure for each prediction step
        weighted_distance_min = 0.0
        hidden_infer = self.coeffs_for_dynamics_prediction.unsqueeze(0).repeat(self.sampling_nbr, 1, 1).view(self.sampling_nbr * num_envs, -1)
        if self.is_circle:
            abs_robot_pos = torch.abs(robot_xy_predictions)
            distance2bounds = torch.max(abs_robot_pos, dim=-1).values
            min_indices = torch.argmin(distance2bounds, dim=0)
            distance2bound = distance2bounds[min_indices, torch.arange(len(min_indices))]
            safe_mask = distance2bound < self.range_limit - range_limit_adjusted_threshold        
            safe_mask = safe_mask.detach().cpu().numpy().astype(bool)
            circle = circle[:, np.newaxis, :]
            vectorized_circle = torch.from_numpy(np.tile(circle[np.newaxis, :, :, :], (self.sampling_nbr, 1, 1, 1))).float().to(device) if self.is_circle else torch.inf            
            # return safe_mask.to(device), min_indices.to(device), distance2bound.to(device)
            if early_return:
                return torch.tensor(safe_mask).to(device), min_indices.to(device), distance2bound.to(device)
            
        else:
            vectorized_buttons = torch.from_numpy(np.tile(buttons[np.newaxis, :, :, :], (self.sampling_nbr, 1, 1, 1))).float().to(device) if self.is_buttons else torch.inf
            vectorized_hazards = torch.from_numpy(np.tile(hazards[np.newaxis, :, :, :], (self.sampling_nbr, 1, 1, 1))).float().to(device) if self.is_hazards else torch.inf
            vectorized_pillars = torch.from_numpy(np.tile(pillars[np.newaxis, :, :, :], (self.sampling_nbr, 1, 1, 1))).float().to(device) if self.is_pillars else torch.inf
            vectorized_goal = torch.from_numpy(np.tile(goal_pos[np.newaxis, :, :, :], (self.sampling_nbr, 1, 1, 1))).float().to(device) if self.is_goal else torch.inf    
            if len(push_boxes.shape) == 2:
                push_boxes = push_boxes[:, np.newaxis, :]

            vectorized_push_boxes = torch.from_numpy(np.tile(push_boxes[np.newaxis, :, :, :], (self.sampling_nbr, 1, 1, 1))).float().to(device) if self.is_push_boxes else torch.inf
            safe_mask = np.ones((self.sampling_nbr, num_envs)).astype(bool)


        for i in range(self.prediction_horizon):
            if i > 0:
                features_for_robot = robot_predictions[:, :, self.robot_slices].reshape(self.sampling_nbr * num_envs, -1)
                input_for_policy = features_for_robot
                reshaped_robot_xyz_predictions = robot_xyz_predictions.reshape(self.sampling_nbr * num_envs, 3).detach().cpu()
                reshaped_robot_matrix_predictions = robot_matrix_predictions.reshape(self.sampling_nbr * num_envs, 3, 3).detach().cpu()

                if self.is_buttons:
                    reshaped_vectorized_buttons = vectorized_buttons.reshape(self.sampling_nbr * num_envs, -1, 3).detach().cpu()
                    features_for_buttons = self._obs_lidar_pseudo(reshaped_vectorized_buttons, reshaped_robot_xyz_predictions, reshaped_robot_matrix_predictions).to(device)
                    input_for_policy = torch.cat([input_for_policy, features_for_buttons], dim=-1)
        
                if self.is_goal:
                    reshaped_vectorized_goal = vectorized_goal.reshape(self.sampling_nbr * num_envs, -1, 3).detach().cpu()
                    features_for_goal = self._obs_lidar_pseudo(reshaped_vectorized_goal, reshaped_robot_xyz_predictions, reshaped_robot_matrix_predictions).to(device)
                    input_for_policy = torch.cat([input_for_policy, features_for_goal], dim=-1)
                    
                if self.is_hazards:
                    # Reshape to match the input shape of _obs_lidar_pseudo
                    reshaped_vectorized_hazards = vectorized_hazards.reshape(self.sampling_nbr * num_envs, -1, 3).detach().cpu()
                    features_for_hazards = self._obs_lidar_pseudo(reshaped_vectorized_hazards, reshaped_robot_xyz_predictions, reshaped_robot_matrix_predictions).to(device)
                    input_for_policy = torch.cat([input_for_policy, features_for_hazards], dim=-1)
                
                if self.is_pillars:
                    reshaped_vectorized_pillars = vectorized_pillars.reshape(self.sampling_nbr * num_envs, -1, 3).detach().cpu()
                    features_for_pillars = self._obs_lidar_pseudo(reshaped_vectorized_pillars, reshaped_robot_xyz_predictions, reshaped_robot_matrix_predictions).to(device)
                    input_for_policy = torch.cat([input_for_policy, features_for_pillars], dim=-1)

                if self.is_push_boxes:
                    reshaped_vectorized_push_boxs = vectorized_push_boxes.reshape(self.sampling_nbr * num_envs, -1, 3).detach().cpu()
                    features_for_push_boxs = self._obs_lidar_pseudo(reshaped_vectorized_push_boxs, reshaped_robot_xyz_predictions, reshaped_robot_matrix_predictions).to(device)
                    input_for_policy = torch.cat([input_for_policy, features_for_push_boxs], dim=-1)

                if self.is_gremlins:
                    # shape of vectorized gremlins: (sampling_nbr, vector_env_nums, mo_nbr, 2)
                    # vector2gremlins = (vectorized_gremlins - robot_pos_predictions[:, :, np.newaxis, :]).reshape(self.sampling_nbr, self.vector_env_nums, -1)
                    gremlins_pos = vectorized_gremlins.reshape(self.sampling_nbr, num_envs, -1)
                    robot_prediction_pos = robot_xy_predictions.reshape(self.sampling_nbr, num_envs, -1)
                    # 2 + 9, rotation matrix is 3 x 3 dimension
                    robot_prediction_mat = robot_matrix_predictions.reshape(self.sampling_nbr, num_envs, -1)
                    sensor_input = torch.cat([robot_prediction_pos, robot_prediction_mat, gremlins_pos], dim=-1)
                    encoded_sensor = self.sensor_predictor.predict(sensor_input)
                    input_for_policy = torch.cat([input_for_policy, encoded_sensor], dim=-1)
                
                if self.is_circle:
                    reshaped_vectorized_circle = vectorized_circle.reshape(self.sampling_nbr * num_envs, -1, 3).detach().cpu()
                    features_for_circle = self._obs_lidar_pseudo(reshaped_vectorized_circle, reshaped_robot_xyz_predictions, reshaped_robot_matrix_predictions).to(device)
                    input_for_policy = torch.cat([input_for_policy, features_for_circle], dim=-1)
                
                input_for_policy = input_for_policy.reshape(self.sampling_nbr * num_envs, -1)
                input_for_policy = torch.cat([input_for_policy, hidden_infer], dim=-1)
                
                input_for_policy = normalizer.normalize(input_for_policy.float())
                # later_actions, _, _, _, _ = policy(input_for_policy)      
                policy_output = policy(input_for_policy)
                if isinstance(policy_output, tuple):
                    later_actions = policy_output[0]
                else:
                    later_actions = policy_output
                
                later_actions = later_actions.reshape(self.sampling_nbr, num_envs, -1).detach().to(device)

                
                obs_dp_input= torch.cat([robot_predictions, later_actions], dim=-1).permute(1, 0, 2)
                
                with torch.no_grad():
                    robot_delta_predictions = self.dynamic_predictor.predict(obs_dp_input, self.coeffs_for_dynamics_prediction).permute(1, 0, 2)
                
                robot_predictions = robot_predictions + robot_delta_predictions
                
                robot_xy_predictions = robot_predictions[:, :, -12:-10]
                robot_xyz_predictions = robot_predictions[:, :, -12: -9]
                robot_matrix_predictions = robot_predictions[:, :, -9:]

            if self.is_gremlins:
                vectorized_gremlins_at_step = vectorized_gremlins[:, :, :, i, :]
                

            if self.is_circle:
                abs_robot_pos = torch.abs(robot_xy_predictions)
                distance2bound, distance2bounds_indices = torch.max(abs_robot_pos, dim=-1)
                distance2bound = distance2bound.detach().cpu().numpy()
                unsafe_mask = distance2bound > self.range_limit - range_limit_adjusted_threshold        
                
                safe_mask = safe_mask & ~unsafe_mask
                distance_min = distance2bound
            else:
                distance2gremlins = compute_min_distance(vectorized_gremlins_at_step[:, :, :, :2], robot_xy_predictions).detach().cpu().numpy() if self.is_gremlins else np.inf
                distance2hazards = compute_min_distance(vectorized_hazards[:, :, :, :2], robot_xy_predictions).detach().cpu().numpy() if self.is_hazards else np.inf
                distance2pillars = compute_min_distance(vectorized_pillars[:, :, :, :2], robot_xy_predictions).detach().cpu().numpy() if self.is_pillars else np.inf

                distance2static = np.minimum(distance2pillars, distance2hazards)
                distance_min = np.minimum(distance2gremlins, distance2static)

                unsafe_mask = np.logical_or(distance2gremlins <= mo_adjusted_threshold, distance2static <= static_adjusted_threshold)
                safe_mask = safe_mask & ~unsafe_mask
            
            weighted_distance_min += 0.9 ** i * distance_min    
            # if one of the vectorized env is unsafe, then stop the prediction
            early_stop = ~np.all(np.all(safe_mask, axis=0))
            if early_stop:
                break

        final_indices = np.zeros(num_envs, dtype=int)

        for env_idx in range(num_envs):
            wdm_env = weighted_distance_min[:, env_idx]
            safe_mask_env = safe_mask[:, env_idx]
            safe_action_indices = np.where(safe_mask_env)[0]

            if len(safe_action_indices) > 0:
                # At least one safe action exists for this environment
                safe_wdm = wdm_env[safe_action_indices]

                if selection_method == 'greedy':
                    best_safe_idx_in_filtered = np.argmax(safe_wdm)
                    selected_idx = safe_action_indices[best_safe_idx_in_filtered]
                elif selection_method == 'top-k':
                    # Ensure k is not larger than the number of safe actions
                    actual_k = min(k, len(safe_action_indices))
                    if actual_k <= 0:
                         # Fallback to greedy if k is 0 or less (should not happen with len > 0 check, but defensive)
                         best_safe_idx_in_filtered = np.argmax(safe_wdm)
                         selected_idx = safe_action_indices[best_safe_idx_in_filtered]
                    else:
                        # Find the indices of the top k distances among safe actions
                        top_k_indices_in_filtered = np.argsort(safe_wdm)[-actual_k:]
                        # Randomly choose one index from the top k
                        chosen_top_k_idx = np.random.choice(top_k_indices_in_filtered)
                        # Map back to the original action index
                        selected_idx = safe_action_indices[chosen_top_k_idx]
                else:
                    # Fallback to greedy as a safety measure
                    best_safe_idx_in_filtered = np.argmax(safe_wdm)
                    selected_idx = safe_action_indices[best_safe_idx_in_filtered]

            else:
                # No safe actions found, choose the action with the highest score (least unsafe)
                selected_idx = np.argmax(wdm_env)

            final_indices[env_idx] = selected_idx

        # Assign the computed indices to max_indices for the return statement
        max_indices = final_indices

        return torch.tensor(safe_mask).to(device), torch.from_numpy(max_indices).to(device), weighted_distance_min