"""
This is training endpoint.
@author
- Van Tuan Nguyen (vantuan.nguyen@lqdtu.edu.vn)
- Razvan Beuran (razvan@jaist.ac.jp)
@create date 2023-12-11 00:28:29
"""

from tqdm import tqdm
import torch
from torch import nn
import pickle
import os
import copy
import numpy as np
from collections import defaultdict
from .model_verifier import ModelVerifier
from Model import Shrink_Autoencoder, Autoencoder

import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ClientTrainer(object):
    """
    Class for training a client model.

    Args:
        model (nn.Module): The model to train.
        loss_function (nn.Module, optional): The loss function to use. Defaults to nn.MSELoss.
        optimizer (torch.optim.Optimizer, optional): The optimizer to use. Defaults to torch.optim.Adam.
        epoch (int, optional): The number of epochs to train for. Defaults to 10.
        batch_size (int, optional): The batch size for training. Defaults to 100.
        lr_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
        patience (int, optional): The number of epochs to wait for improvement in validation loss before early stopping. Defaults to 5.
        save_dir (str, optional): The directory to save the trained model and training tracking information. Defaults to "Checkpoint/ClientModel/".
        fedprox_mu (float, optional): The FedProx regularization parameter. Defaults to 0.001.
        client_id (str, optional): The ID of the client. Defaults to None.
        model_type (str, optional): The type of the model. Defaults to "hybrid".
        verification_method (str, optional): The method for model verification. Defaults to "val".
        verification_threshold (float, optional): The threshold for verification. Defaults to 3.0.
        performance_threshold (float, optional): The threshold for performance. Defaults to 0.002.
    """

    def __init__(self, model=None, loss_function=nn.MSELoss, optimizer=torch.optim.Adam,
                    epoch=10, batch_size=100, lr_rate=1e-3, update_type="avg",
                    patience=5, save_dir="Checkpoint/ClientModel/", fedprox_mu=0.001, 
                    client_id=None, model_type="hybrid", verification_method="val", 
                    verification_threshold=3.0, performance_threshold=0.002) -> None:
        
        if model is None:
            logging.info(f"[Client {client_id}] Have to indicate the model to train.")
            return None
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logging.info(f"[Client {client_id}] Created saving dir.")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.previous_global_model = copy.deepcopy(self.model)
        self.loss_function = loss_function
        self.lr_rate = lr_rate
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr_rate)
        self.epoch = epoch
        self.batch_size = batch_size
        self.patience = patience
        self.save_dir = save_dir
        self.update_type = update_type
        self.fedprox_mu = fedprox_mu
        self.client_id = client_id
        
        # New attributes for peer-to-peer communication
        self.peers = []  # List of peer clients
        self.aggregation_count = 0
        self.max_aggregation_threshold = 3
        self.mse_score = float('inf')
        self.votes_received = 0
        self.current_round = -1
        self.has_aggregated_this_round = False
        self.received_models = {}  # Store received model updates from peers
        self.validation_data = None  # Store validation data for aggregation
        self.dev_dataset = None  # Store development dataset for aggregation
        
        # Add verification components
        self.verifier = ModelVerifier(
            verification_threshold=verification_threshold,
            performance_threshold=performance_threshold,
            verification_method=verification_method
        )
        self.rejected_updates = 0
        self.max_rejected_updates = 3
        self.model_type = model_type  # Use the passed model_type parameter

    def create_dev_dataset(self, dataset):
        """Create development dataset for aggregation"""
        if isinstance(dataset["dataset"], np.ndarray):
            self.dev_dataset = torch.Tensor(dataset["dataset"])
        else:
            self.dev_dataset = dataset["dataset"]
        # Set the development dataset in the verifier
        self.verifier.set_dev_dataset(self.dev_dataset)
        logging.info(f"[Client {self.client_id}] Created development dataset for aggregation and verification")

    def fed_avg(self, local_models):
        """Perform federated averaging"""
        total_samples = sum(num_samples for _, num_samples in local_models)
        avg_weights = {}
        for key in local_models[0][0].keys():
            avg_weights[key] = sum(model[0][key] * (model[1] / total_samples) for model in local_models)
        return avg_weights

    def fed_mse_avg(self, local_models):
        """Perform MSE-based weighted averaging"""
        update_weights = []
        for model_state, num_samples in local_models:
            self.model.load_state_dict(model_state)
            self.model.eval()
            with torch.no_grad():
                _, generated_data, _ = self.model(torch.Tensor(self.dev_dataset).to(self.device))
                mse_score = torch.nn.MSELoss(reduction='mean')(torch.Tensor(self.dev_dataset).to(self.device), generated_data)
                update_weights.append((model_state, 1/mse_score))
        
        avg_weights = {}
        total_weight = sum(weight for _, weight in update_weights)
        for key in update_weights[0][0].keys():
            avg_weights[key] = sum(w[key] * (weight/total_weight) for w, weight in update_weights)
        return avg_weights

    def fedprox(self, local_models):
        """Perform FedProx aggregation"""
        return self.fed_avg(local_models)  # FedProx uses same averaging but adds proximal term in training

    def connect_to_peers(self, peers):
        """Connect to peer clients for P2P communication"""
        self.peers = peers
        logging.info(f"[Client {self.client_id}] Connected to {len(peers)} peers")

    def broadcast_model(self):
        """Broadcast current model to all peers"""
        model_state = self.model.state_dict()
        for peer in self.peers:
            peer.receive_model(self, model_state)
        logging.info(f"[Client {self.client_id}] Model broadcasted to all peers")

    def receive_model(self, sender, model_state):
        """Receive model update from a peer"""
        self.received_models[sender] = model_state
        logging.info(f"[Client {self.client_id}] Received model from peer {sender}")

    def request_aggregation(self):
        """Request aggregation from peers"""
        if self.aggregation_count >= self.max_aggregation_threshold or not self.received_models:
            return None
        
        local_models = [(model_state, 1) for model_state in self.received_models.values()]
        
        if self.update_type == "avg":
            aggregated_state = self.fed_avg(local_models)
        elif self.update_type == "mse_avg":
            aggregated_state = self.fed_mse_avg(local_models)
        elif self.update_type == "fedprox":
            aggregated_state = self.fedprox(local_models)
        else:
            logging.error(f"[Client {self.client_id}] Unknown update type: {self.update_type}")
            return None
        
        self.aggregation_count += 1
        self.has_aggregated_this_round = True
        return aggregated_state

    def update_from_peers(self):
        """Update model based on received peer updates with verification"""
        if not self.received_models:
            return
        
        # Get the aggregated model from the aggregator
        aggregated_state = list(self.received_models.values())[0]  # Should only have one model from aggregator
        
        # Verify the aggregated model
        is_verified, performance_change = self.verifier.verify_model(
            self.client_id,
            aggregated_state,
            self.validation_data,
            self.current_round,
            self.model_type
        )
        
        if is_verified:
            self.model.load_state_dict(aggregated_state)
            self.previous_global_model = copy.deepcopy(self.model)
            self.rejected_updates = 0  # Reset counter on successful update
            logging.info(f"[Client {self.client_id}] Model verified and updated. Performance change: {performance_change:.10f}")  # Performance change: {performance_change:.10f}"
            
        else:
            self.rejected_updates += 1
            logging.warning(f"[Client {self.client_id}] Model update rejected. Performance change: {performance_change:.10f}")
            
            if self.rejected_updates >= self.max_rejected_updates:
                logging.error(f"[Client {self.client_id}] Too many rejected updates. Possible attack detected.")
                # Implement additional security measures here
        
        # Clear received models after update attempt
        self.received_models.clear()

    def calculate_mse_score(self, validation_data):
        """
        Calculate MSE score for voting mechanism.
        
        Args:
            validation_data (torch.Tensor): Validation data to calculate MSE on.
            
        Returns:
            float: Normalized MSE score
        """
        self.model.eval()
        
        # Normalize validation data
        mean = validation_data.mean(dim=0, keepdim=True)
        std = validation_data.std(dim=0, keepdim=True) + 1e-8  # Add small epsilon to avoid division by zero
        normalized_data = (validation_data - mean) / std
        
        # Process in batches to handle large datasets
        batch_size = 128
        total_mse = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(normalized_data), batch_size):
                batch = normalized_data[i:i + batch_size].to(self.device)
                _, generated_data, _ = self.model(batch)
                
                # Calculate MSE for this batch
                batch_mse = torch.nn.MSELoss(reduction='mean')(batch, generated_data)
                total_mse += batch_mse.item()
                num_batches += 1
        
        # Calculate average MSE across all batches
        avg_mse = total_mse / num_batches if num_batches > 0 else float('inf')
        
        # Add a very small random component to break ties (0.01% variation)
        random_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.0002
        self.mse_score = avg_mse * random_factor
        
        return self.mse_score

    def vote_for_aggregator(self, clients, validation_data, current_round):
        """
        Vote for the best aggregator based on MSE scores.
        
        Args:
            clients (list): List of ClientTrainer instances
            validation_data (torch.Tensor): Validation data to calculate MSE on
            current_round (int): Current round number
            
        Returns:
            ClientTrainer: Selected aggregator or None if no valid aggregator found
        """
        # Update current round for all clients
        for client in clients:
            client.current_round = current_round
            client.has_aggregated_this_round = False
        
        # Calculate MSE scores for all clients
        mse_scores = []
        for client in clients:
            if client != self:  # Don't vote for self
                mse_score = client.calculate_mse_score(validation_data)
                mse_scores.append((client, mse_score))
                logging.info(f"[Client {self.client_id}] Client {clients.index(client) + 1} MSE score: {mse_score:.6f}")
        
        # Sort by MSE score (lower is better)
        mse_scores.sort(key=lambda x: x[1])
        
        # Vote for the client with lowest MSE that hasn't exceeded aggregation threshold
        for client, mse_score in mse_scores:
            if client.aggregation_count < client.max_aggregation_threshold:
                client.votes_received += 1
                client_index = clients.index(client) + 1
                logging.info(f"[Client {self.client_id}] Voting for Client {client_index} with MSE score: {mse_score:.6f}")
                return client
        
        return None
    
    def aggregate_models(self, clients, validation_data, current_round):
        """
        Aggregate models from all clients using the specified update type.
        
        Args:
            clients (list): List of ClientTrainer instances
            validation_data (torch.Tensor): Validation data to calculate MSE on
            current_round (int): Current round number
            
        Returns:
            dict: Aggregated model state dict or None if aggregation not possible
        """
        # Check if this client should perform aggregation
        if (self.aggregation_count >= self.max_aggregation_threshold or 
            self.has_aggregated_this_round):
            logging.warning(f"[Client {self.client_id}] This client cannot perform aggregation in this round")
            return None
            
        # Collect all client models
        local_models = []
        for client in clients:
            model_state = client.model.state_dict()
            if self.update_type == "mse_avg":
                # Calculate weight based on MSE score
                mse_score = client.calculate_mse_score(validation_data)
                weight = 1.0 / (mse_score + 1e-10)
            else:
                weight = 1.0  # Equal weight for avg and fedprox
            local_models.append((model_state, weight))
        
        # Perform aggregation based on update type
        if self.update_type == "avg":
            aggregated_state = self.fed_avg(local_models)
        elif self.update_type == "mse_avg":
            aggregated_state = self.fed_mse_avg(local_models)
        elif self.update_type == "fedprox":
            aggregated_state = self.fedprox(local_models)
        else:
            logging.error(f"[Client {self.client_id}] Unknown update type: {self.update_type}")
            return None
        
        # Update aggregation tracking
        self.aggregation_count += 1
        self.has_aggregated_this_round = True
        
        # Update this client's model with the aggregated state
        self.model.load_state_dict(aggregated_state)
        
        return aggregated_state
    
    def save_model(self):
        """
        Save the trained model to the specified directory.
        """
        logging.info(f"[Client {self.client_id}] Saving model to {self.save_dir}")
        save_file = os.path.join(self.save_dir, "model.cpt")
        try:
            torch.save(
                self.model.state_dict(),
                save_file,
                _use_new_zipfile_serialization=False,
            )
        except:
            torch.save(self.model.state_dict(), save_file)
    
    
    def save_tracking_information(self):
        """
        Save the training tracking information.
        This method is currently empty and can be implemented as needed.
        """
        pass
    
    def run(self, train_loader, valid_loader=None):
        """Run the training process"""
        min_valid_loss = float("inf")
        worse_count = 0
        training_tracking = []
        
        for epoch in range(self.epoch):
            self.model.train()
            epoch_loss = 0
            # for i, batch_input in zip(tqdm(range(len(train_loader)), desc='Training batch: ...'), train_loader):
            for i, batch_input in enumerate(train_loader):
                _, _, loss = self.model(batch_input[0].to(self.device))
                
                # Add proximal term for FedProx
                if self.update_type == "fedprox":
                    prox_term = 0.0
                    for param, global_param in zip(self.model.parameters(), self.previous_global_model.parameters()):
                        prox_term += torch.sum(torch.square(param - global_param.to(self.device)))
                    loss += self.fedprox_mu * prox_term
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss += loss.item()
            
            epoch_loss = epoch_loss / len(train_loader)
            
            if valid_loader is not None:
                valid_loss = 0
                self.model.eval()
                with torch.no_grad():
                    # for i, batch_input in zip(tqdm(range(len(valid_loader)), desc='Validating batch: ...'), valid_loader):
                    for i, batch_input in enumerate(valid_loader):
                        _, _, loss = self.model(batch_input[0].to(self.device))
                        
                        # Add proximal term for FedProx validation
                        if self.update_type == "fedprox":
                            prox_term = 0.0
                            for param, global_param in zip(self.model.parameters(), self.previous_global_model.parameters()):
                                prox_term += torch.sum(torch.square(param - global_param.to(self.device)))
                            loss += self.fedprox_mu * prox_term
                        
                        valid_loss += loss.item()
                    
                    valid_loss = valid_loss / len(valid_loader)
                    training_tracking.append((epoch_loss, valid_loss))
                    logging.info(f"[Client {self.client_id}] Epoch {epoch+1} - Training loss: {epoch_loss} - Validating loss: {valid_loss}")

                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    self.save_model()
                    worse_count = 0
                else:
                    worse_count += 1
                    if worse_count >= self.patience:
                        logging.info(f"[Client {self.client_id}] Early stopping in epoch {epoch+1}.")
                        pickle.dump(training_tracking, open(os.path.join(self.save_dir, "training_tracking.pkl"), "wb"))
                        break
            
            pickle.dump(training_tracking, open(os.path.join(self.save_dir, "training_tracking.pkl"), "wb"))
            