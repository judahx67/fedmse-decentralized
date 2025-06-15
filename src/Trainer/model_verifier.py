"""
Model verification system for federated learning.
@author
- Van Tuan Nguyen (vantuan.nguyen@lqdtu.edu.vn)
- Razvan Beuran (razvan@jaist.ac.jp)
"""

import torch
import logging
import numpy as np
from Model import Autoencoder, Shrink_Autoencoder

class ModelVerifier:
    def __init__(self, verification_threshold=3.0, performance_threshold=0.002):
        self.verification_threshold = verification_threshold
        self.performance_threshold = performance_threshold
        self.history = {}  # Store historical model states and performance
        
    def verify_model(self, client_id, new_model_state, validation_data, current_round, model_type="hybrid"):
        """
        Verify if the received aggregated model is acceptable
        Returns: (bool, float) - (is_verified, performance_change)
        """
        if client_id not in self.history:
            self.history[client_id] = {
                'model_state': new_model_state,
                'performance': self._evaluate_model(new_model_state, validation_data, model_type),
                'round': current_round
            }
            return True, 0.0
            
        # Get previous state
        prev_state = self.history[client_id]['model_state']
        prev_performance = self.history[client_id]['performance']
        
        # Check parameter changes
        param_changes = self._calculate_parameter_changes(prev_state, new_model_state)
        
        # Evaluate new model performance
        new_performance = self._evaluate_model(new_model_state, validation_data, model_type)
        performance_change = new_performance - prev_performance
        
        # Update history
        self.history[client_id] = {
            'model_state': new_model_state,
            'performance': new_performance,
            'round': current_round
        }
        
        # Log the actual values for debugging
        logging.info(f"Client {client_id} - Param changes: {param_changes:.6f}, Performance change: {performance_change:.6f}")
        
        # Verify based on both parameter changes and performance
        is_verified = (
            param_changes <= self.verification_threshold and
            performance_change >= -self.performance_threshold
        )
        
        return is_verified, performance_change
        
    def _calculate_parameter_changes(self, old_state, new_state):
        """Calculate the magnitude of parameter changes"""
        total_change = 0
        for key in old_state.keys():
            change = torch.norm(old_state[key] - new_state[key])
            total_change += change
        return total_change
        
    def _evaluate_model(self, model_state, validation_data, model_type):
        """Evaluate model performance on validation data"""
        # Create appropriate model based on type
        if model_type == "hybrid":
            model = Shrink_Autoencoder(input_dim=validation_data.shape[1], output_dim=validation_data.shape[1])
        else:
            model = Autoencoder(input_dim=validation_data.shape[1], output_dim=validation_data.shape[1])
            
        model.load_state_dict(model_state)
        model.eval()
        
        with torch.no_grad():
            _, generated_data, _ = model(validation_data)
            mse = torch.nn.MSELoss()(validation_data, generated_data)
            return 1.0 / (1.0 + mse)  # Convert to a score where higher is better 