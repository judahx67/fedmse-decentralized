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
import networkx as nx
import random
import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class GlobalAggregator(object):
    def __init__(self, model, update_type="avg", topology_type="ring", num_clients=10):
        """
            - initialize the SAE model in global: the model architecture: input-dim, 
                output-dim, latent-dim
            - 
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_type = update_type
        self.model = model.to(self.device)
        self.num_clients = num_clients
        self.topology_type = topology_type
        self.topology = self._create_topology()
        self._log_topology_info()
        
    def _log_topology_info(self):
        """Log information about the network topology"""
        logging.info(f"\n{'='*50}")
        logging.info(f"Network Topology Information:")
        logging.info(f"Type: {self.topology_type}")
        logging.info(f"Number of nodes: {self.num_clients}")
        logging.info(f"Average node degree: {sum(dict(self.topology.degree()).values())/self.num_clients:.2f}")
        logging.info(f"Network diameter: {nx.diameter(self.topology)}")
        logging.info(f"Network density: {nx.density(self.topology):.3f}")
        logging.info(f"{'='*50}\n")
        
    def _create_topology(self):
        """
        Create the network topology for decentralized communication
        """
        if self.topology_type == "ring":
            # Create a ring topology
            G = nx.Graph()
            for i in range(self.num_clients):
                G.add_edge(i, (i + 1) % self.num_clients)
        elif self.topology_type == "random":
            # Create a random topology with average degree of 3
            G = nx.Graph()
            for i in range(self.num_clients):
                G.add_node(i)
            # Add random edges
            for i in range(self.num_clients):
                num_edges = random.randint(2, 4)  # Random number of connections
                for _ in range(num_edges):
                    j = random.randint(0, self.num_clients - 1)
                    if i != j:
                        G.add_edge(i, j)
        elif self.topology_type == "mesh":
            # Create a mesh topology (fully connected)
            G = nx.complete_graph(self.num_clients)
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
        
        return G

    def decentralized_update(self, local_models, num_rounds=5):
        """
        Perform decentralized federated learning updates
        
        Args:
            local_models (list): List of local models
            num_rounds (int): Number of communication rounds
            
        Returns:
            list: Updated local models
        """
        logging.info(f"\n{'='*50}")
        logging.info(f"Starting Decentralized Learning")
        logging.info(f"Topology: {self.topology_type}")
        logging.info(f"Number of rounds: {num_rounds}")
        logging.info(f"Number of models: {len(local_models)}")
        logging.info(f"{'='*50}\n")
        
        # Extract just the model state dicts from the tuples
        updated_models = [model[0] for model in local_models]
        num_models = len(updated_models)
        
        # Create a temporary topology for this update
        temp_topology = self._create_topology()
        # Ensure topology has correct number of nodes
        if len(temp_topology.nodes()) != num_models:
            temp_topology = nx.Graph()
            for i in range(num_models):
                temp_topology.add_node(i)
            # Add edges based on topology type
            if self.topology_type == "ring":
                for i in range(num_models):
                    temp_topology.add_edge(i, (i + 1) % num_models)
            elif self.topology_type == "random":
                for i in range(num_models):
                    num_edges = random.randint(2, min(4, num_models-1))
                    for _ in range(num_edges):
                        j = random.randint(0, num_models - 1)
                        if i != j:
                            temp_topology.add_edge(i, j)
            elif self.topology_type == "mesh":
                for i in range(num_models):
                    for j in range(i+1, num_models):
                        temp_topology.add_edge(i, j)
        
        for round in range(num_rounds):
            round_communications = 0
            round_aggregations = 0
            
            # For each client, aggregate with its neighbors
            for client_id in range(num_models):
                # Get neighbors of current client
                neighbors = list(temp_topology.neighbors(client_id))
                
                if not neighbors:
                    continue
                
                round_communications += len(neighbors)
                round_aggregations += 1
                
                # Collect models from neighbors
                neighbor_models = [updated_models[neighbor] for neighbor in neighbors]
                neighbor_models.append(updated_models[client_id])  # Include own model
                
                # Calculate weights based on number of samples
                total_samples = sum(local_models[j][2] for j in neighbors + [client_id])
                weights = [local_models[j][2] / total_samples for j in neighbors + [client_id]]
                
                # Aggregate models
                avg_weights = {}
                for key in neighbor_models[0].keys():
                    avg_weights[key] = sum(model[key] * weight 
                                         for model, weight in zip(neighbor_models, weights))
                
                # Update client's model
                updated_models[client_id] = avg_weights
            
            logging.info(f"\nRound {round + 1}/{num_rounds} Statistics:")
            logging.info(f"Total communications: {round_communications}")
            logging.info(f"Total aggregations: {round_aggregations}")
            logging.info(f"Average communications per node: {round_communications/num_models:.2f}")
        
        logging.info(f"\n{'='*50}")
        logging.info(f"Decentralized Learning Complete")
        logging.info(f"Total rounds: {num_rounds}")
        logging.info(f"Final topology: {self.topology_type}")
        logging.info(f"{'='*50}\n")
        
        # Convert back to the original format with sample counts
        return [(model, local_models[i][1], local_models[i][2]) for i, model in enumerate(updated_models)]

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
    
    def update(self, local_models=None):
        """
        Update the global model using the local models.

        Args:
            local_models (list): List of local models to be used for updating the global model.

        Returns:
            None
        """
        if self.update_type == "decentralized":
            updated_models = self.decentralized_update(local_models)
            # Use the last client's model as the global model
            self.model.load_state_dict(updated_models[-1][0])
        elif self.update_type == "avg":
            self.fed_avg(local_models)
        elif self.update_type == "fusion_avg":
            self.fusion_avg(local_models)
        elif self.update_type == "mse_avg":
            self.fed_mse_avg(local_models)
        elif self.update_type == "fedprox":
            self.fedprox(local_models, mu=0.01)
        
        self.model.eval()
        with torch.no_grad():
            _, _, val_loss = self.model(torch.Tensor(self.dev_dataset).to(self.device))
            self.val_loss = val_loss.item()
    
    


