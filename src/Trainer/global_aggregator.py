"""
This is the global model update function.
@author
- Van Tuan Nguyen (vantuan.nguyen@lqdtu.edu.vn)
- Razvan Beuran (razvan@jaist.ac.jp)
@create date 2023-12-14 11:41:08
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from Utils import similarity_score
from DataLoader import IoTDataProccessor
import logging
import copy

# Configure the logging module
logging.basicConfig(level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(levelname)s - %(message)s')

class GlobalAggregator(object):
    def __init__(self, model, update_type="avg"):
        """
        Initialize the global aggregator.
        
        Args:
            model: The model architecture to use
            update_type (str): Type of update mechanism to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_type = update_type
        self.model = model.to(self.device)
        self.aggregation_history = {}  # Track aggregation history for each client
    
    def select_aggregator(self, clients, validation_data):
        """
        Select the best client to perform aggregation based on voting.
        
        Args:
            clients (list): List of ClientTrainer instances
            validation_data (torch.Tensor): Validation data for MSE calculation
            
        Returns:
            ClientTrainer: Selected aggregator client
        """
        logging.info("Starting aggregator selection process...")
        
        # Reset votes for this round
        for client in clients:
            client.votes_received = 0
        
        # Each client votes for the best aggregator
        for i, client in enumerate(clients):
            logging.info(f"Client {i+1} is voting...")
            selected_aggregator = client.vote_for_aggregator(clients, validation_data)
            if selected_aggregator:
                self.aggregation_history[selected_aggregator] = self.aggregation_history.get(selected_aggregator, 0) + 1
                logging.info(f"Client {i+1} voted for aggregator with MSE score: {selected_aggregator.mse_score:.4f}")
        
        # Select client with most votes that hasn't exceeded threshold
        best_client = None
        max_votes = -1
        
        for i, client in enumerate(clients):
            logging.info(f"Client {i+1} received {client.votes_received} votes and has been aggregator {client.aggregation_count} times")
            if (client.votes_received > max_votes and 
                client.aggregation_count < client.max_aggregation_threshold):
                max_votes = client.votes_received
                best_client = client
        
        if best_client:
            logging.info(f"Selected aggregator with {max_votes} votes and MSE score: {best_client.mse_score:.4f}")
        else:
            logging.warning("No suitable aggregator found")
        
        return best_client
    
    def update(self, clients, validation_data):
        """
        Update the global model using decentralized aggregation.
        
        Args:
            clients (list): List of ClientTrainer instances
            validation_data (torch.Tensor): Validation data for MSE calculation
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        logging.info("Starting decentralized model update...")
        
        # Select aggregator
        aggregator = self.select_aggregator(clients, validation_data)
        if not aggregator:
            logging.warning("No suitable aggregator found")
            return False
        
        logging.info("Performing model aggregation...")
        # Perform aggregation
        aggregated_state = aggregator.aggregate_models(clients, validation_data)
        if aggregated_state is None:
            logging.warning("Aggregation failed")
            return False
        
        # Update global model
        self.model.load_state_dict(aggregated_state)
        
        # Update all clients with new global model
        for client in clients:
            client.model.load_state_dict(aggregated_state)
            client.previous_global_model = copy.deepcopy(client.model)
        
        logging.info("Model update completed successfully")
        return True
    
    def create_dev_dataset(self, dataset):
        """
        Choose a development dataset for updating the global model (Fusion-based)
        
        params: 
            - dataset: (pytorch Dataset) the dataset concatenated from clients's dev sets.
        """
        logging.info("Creating development dataset...")
        self.dev_dataset = dataset["dataset"]
        # self.dev_label = dataset["label"]
        data_processor = IoTDataProccessor(scaler="standard")
        
        self.dev_dataset, self.dev_label = data_processor.fit_transform(self.dev_dataset)
        
        if self.update_type == "fusion_avg":
            self.dev_kde_scores = KernelDensity(kernel='gaussian', bandwidth="scott") \
                .fit(self.dev_dataset).score_samples(self.dev_dataset)
            
    def fusion_avg(self, local_models=None):
        """
        Perform fusion-based updating by calculating the average weights of local models.

        Args:
            local_models (list): List of local models.

        Returns:
            None
        """
        logging.info("Fusion-based updating...")
        update_weights = []
        weighted = []
        for i, local_model in zip(tqdm(range(len(local_models)), desc='Calculating similarity...'), local_models):
            self.model.load_state_dict(local_model)
            self.model.eval()
            with torch.no_grad():
                _, generated_data, _ = self.model(torch.Tensor(self.dev_dataset).to(self.device))
                sim_score = similarity_score(self.dev_kde_scores, generated_data.cpu().numpy())    # Calculate similarity score
                weighted.append(1/sim_score)    # Because sim_score close to 0 when more similar, so we use 1/sim_score
                update_weights.append((local_model, 1/sim_score))
        print(weighted)
        avg_weights = {}
        for key in update_weights[0][0].keys():
            avg_weights[key] = sum([w[key] * alpha  for w, alpha in update_weights]) \
                / sum([alpha for w, alpha in update_weights])
        self.model.load_state_dict(avg_weights)
        
    def fed_mse_avg(self, local_models=None):
        """
        Perform fusion-based updating by calculating the average weights of local models
        based on MSE loss of each AE-based model.

        Args:
            local_models (list): List of local models.

        Returns:
            None
        """
        logging.info("MSE-based updating...")
        update_weights = []
        weighted = []
        for i, local_model in zip(tqdm(range(len(local_models)), desc='Calculating similarity...'), local_models):
            self.model.load_state_dict(local_model[0])
            self.model.eval()
            with torch.no_grad():
                _, generated_data, _ = self.model(torch.Tensor(self.dev_dataset).to(self.device))
                sim_score = torch.nn.MSELoss(reduction='mean')(torch.Tensor(self.dev_dataset).to(self.device), generated_data)  # Calculate similarity score
                weighted.append(1/sim_score)    # Because sim_score close to 0 when more similar, so we use 1/sim_score
                update_weights.append((local_model[0], 1/sim_score))
        avg_weights = {}
        for key in update_weights[0][0].keys():
            avg_weights[key] = sum([w[key] * alpha  for w, alpha in update_weights]) \
                / sum([alpha for w, alpha in update_weights])
                
        self.model.load_state_dict(avg_weights)
    
    def fed_avg(self, local_models=None):
        """
        Perform federated averaging to aggregate the weights of local models.

        Args:
            local_models (list): List of dictionaries representing the weights of local models.

        Returns:
            None
        """
        # avg_weights = {}
        # for key in local_models[0].keys():
        #     avg_weights[key] = sum([w[key] for w in local_models]) / len(local_models)
        
        # self.model.load_state_dict(avg_weights)
        
        # Sum the total number of samples across all clients
        total_samples = sum(model[2] for model in local_models)
        
        # Initialize the averaged weights dictionary
        avg_weights = {}

        # Iterate over each key in the state_dict of the first local model
        for key in local_models[0][0].keys():
            # Initialize the key in avg_weights
            avg_weights[key] = sum(model[0][key] * (model[2] / total_samples) for model in local_models)

        # Load the averaged weights into the global model
        self.model.load_state_dict(avg_weights)
    
    def fedprox(self, local_models=None, mu=0.01):
        """
        Perform federated optimization using the FedProx algorithm to aggregate the weights of local models.

        Args:
            local_models (list): List of dictionaries representing the weights of local models.
            mu (float): Proximal term coefficient.

        Returns:
            None
        """
        logging.info("FedProx updating...")
        # global_weights = self.model.state_dict()
        # update_weights = []
        # for local_model in local_models:
        #     for key in local_model.keys():
        #         local_model[key] -= mu * (local_model[key] - global_weights[key])
        #     update_weights.append(local_model)
        
        # avg_weights = {}
        # for key in global_weights.keys():
        #     avg_weights[key] = sum([w[key] for w in update_weights]) / len(update_weights)
        
        # self.model.load_state_dict(avg_weights)
        
        # Sum the total number of samples across all clients
        total_samples = sum(model[2] for model in local_models)
        
        # Initialize the averaged weights dictionary
        avg_weights = {}

        # Iterate over each key in the state_dict of the first local model
        for key in local_models[0][0].keys():
            # Initialize the key in avg_weights
            avg_weights[key] = sum(model[0][key] * (model[2] / total_samples) for model in local_models)

        # Load the averaged weights into the global model
        self.model.load_state_dict(avg_weights)
    
