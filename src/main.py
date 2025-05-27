"""
This is training endpoint.
@author
- Van Tuan Nguyen (vantuan.nguyen@lqdtu.edu.vn)
- Razvan Beuran (razvan@jaist.ac.jp)
@create date 2023-12-11 00:28:29
"""

import os
import json
import pickle   
import pandas as pd
import numpy as np
import torch
import argparse
import copy
import random
import json
from torch.utils.data import DataLoader, random_split, ConcatDataset
from Model import Shrink_Autoencoder
from Model import Autoencoder
from DataLoader import load_data
from DataLoader import IoTDataset
from DataLoader import IoTDataProccessor
from Trainer import ClientTrainer
from Trainer import GlobalAggregator
from Evaluator import Evaluator
import networkx as nx
from collections import defaultdict

import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(levelname)s - %(message)s')


num_participants = 0.5
epoch = 1 #100
num_rounds = 1 #20
lr_rate = 1e-5
shrink_lambda = 1 #10
network_size = 10 #50
data_seed = 1111 #1234
topology_type = "ring"  # Options: "ring", "random", "mesh"
# no_Exp = f"nonIID_Exp1_Rerun_{epoch}epoch_10client_lr0001_lamda{shrink_lambda}_ratio{num_participants*100}"
no_Exp = f"IID-Update_Exp6_scale_{epoch}epoch_{network_size}client_{num_rounds}rounds_lr{lr_rate}_lamda{shrink_lambda}_ratio{num_participants*100}_dataseed{data_seed}"

num_runs = 1
batch_size = 12

new_device = True
min_val_loss = float("inf")
global_patience = 1
global_worse = 0
metric = "AUC" #AUC or classification
# model_type = "autoencoder"   #autoencoder; hybrid;
# update_type = "mse_avg"  #avg; fusion_avg; mse_avg
dim_features = 115   #nba-iot: 115; cic-2023: 46

scen_name = 'FL-IoT' 

#config_file = f"Configuration/scen2-nba-iot-10clients.json"
config_file = "Configuration/scen2-nba-iot-10clients.json"
# config_file = "Configuration/cic-config.json"

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_trust_score(model, client_data, device):
    """Calculate trust score for a node based on model performance and data quality"""
    model.eval()
    with torch.no_grad():
        # Calculate reconstruction error on client's data
        _, generated_data, loss = model(client_data.to(device))
        reconstruction_error = loss.item()
        
        # Calculate data quality metrics
        data_quality = np.std(client_data.cpu().numpy())  # Higher variance might indicate better data diversity
        
        # Combine metrics (lower reconstruction error and higher data quality = higher trust)
        trust_score = data_quality / (1 + reconstruction_error)
        
    return trust_score

def select_aggregator(client_models, client_info, device, G):
    """Select the most trusted node as aggregator through voting"""
    trust_scores = []
    votes = defaultdict(int)
    
    # Each node calculates trust scores for all other nodes
    for i, (model, client) in enumerate(zip(client_models, client_info)):
        # Get a sample of data from the client
        sample_data = next(iter(client["train_loader"]))[0]
        
        # Calculate trust score for this node
        trust_score = calculate_trust_score(model, sample_data, device)
        trust_scores.append((i, trust_score))
    
    # Each node votes for the node with highest trust score among its neighbors
    for i in range(len(client_models)):
        neighbors = list(G.neighbors(i))
        if not neighbors:
            continue
            
        # Get trust scores of neighbors
        neighbor_scores = [(j, score) for j, score in trust_scores if j in neighbors]
        if neighbor_scores:
            # Vote for the neighbor with highest trust score
            best_neighbor = max(neighbor_scores, key=lambda x: x[1])[0]
            votes[best_neighbor] += 1
    
    # Select the node with most votes as aggregator
    if votes:
        aggregator = max(votes.items(), key=lambda x: x[1])[0]
        logging.info(f"\nVoting Results:")
        logging.info(f"Trust Scores: {dict(trust_scores)}")
        logging.info(f"Votes: {dict(votes)}")
        logging.info(f"Selected Aggregator: Node {client_info[aggregator]['device']}")
        return aggregator
    else:
        # If no votes, select the node with highest trust score
        aggregator = max(trust_scores, key=lambda x: x[1])[0]
        logging.info(f"\nNo votes cast, selecting node with highest trust score:")
        logging.info(f"Trust Scores: {dict(trust_scores)}")
        logging.info(f"Selected Aggregator: Node {client_info[aggregator]['device']}")
        return aggregator

if __name__ == "__main__":
        random.seed(data_seed)
        np.random.seed(data_seed)
        try:
            logging.info("Loading configuration...")
            with open(config_file, "r") as config_file:
                config = json.load(config_file)
        except Exception as e:
            logging.info("Failed to load configuration.")
        
        devices_list = random.sample(config['devices_list'], network_size)
        # devices_list = config['devices_list']
        client_info = []
        # random.seed(data_seed)
        # np.random.seed(data_seed)
        for device in devices_list:
            logging.info("Creating metadata for client...")
            normal_data_path = os.path.join(config['data_path'], device["normal_data_path"])
            abnormal_data_path = os.path.join(config['data_path'], device["abnormal_data_path"])
            test_new_normal_data_path = os.path.join(config['data_path'], device["test_normal_data_path"])
            
            logging.info("Loading data from {}...".format(device['name']))
            
            # normal_data = load_data(normal_data_path, header="infer")
            normal_data = load_data(normal_data_path)
            normal_data = normal_data.sample(frac=1).reset_index(drop=True)
            # abnormal_data = load_data(abnormal_data_path, header="infer")

            abnormal_data = load_data(abnormal_data_path)
            abnormal_data = abnormal_data.sample(frac=1).reset_index(drop=True)
            
            # new normal data from new devices
            if new_device:
                new_normal_data = load_data(test_new_normal_data_path)
            
            device_name = device['name']
            print(f"{device_name} has {len(normal_data)} normal data and {len(abnormal_data)} abnormal data")
            # now, need to split data before normalization
            train_normal_size = int(0.4 * len(normal_data))
            valid_normal_size = int(0.1 * len(normal_data))
            dev_normal_size = int(0.4 * len(normal_data))
            test_normal_size = len(normal_data) - train_normal_size - valid_normal_size - dev_normal_size
            
            train_normal_data = normal_data[:train_normal_size]
            valid_normal_data = normal_data[train_normal_size:train_normal_size+valid_normal_size]
            dev_normal_data = normal_data[train_normal_size+valid_normal_size:train_normal_size+valid_normal_size+dev_normal_size]
            test_normal_data = normal_data[train_normal_size+valid_normal_size+dev_normal_size:]

            data_processor = IoTDataProccessor(scaler="standard")
            processed_train_data, train_label = data_processor.fit_transform(train_normal_data)
            processed_valid_data, valid_label = data_processor.transform(valid_normal_data)
            # processed_dev_data, dev_label = data_processor.transform(dev_normal_data)
            processed_test_data, test_label = data_processor.transform(test_normal_data)
            processed_abnormal_data, abnormal_label = data_processor.transform(abnormal_data, type="abnormal")
            
            if new_device:
                processed_new_normal_data, new_normal_label = data_processor.transform(new_normal_data)
                processed_test_data = np.concatenate([processed_test_data, processed_new_normal_data], axis=0)
                processed_test_label = np.concatenate([test_label, new_normal_label], axis=0)
                test_dataset = IoTDataset(processed_test_data, processed_test_label)
            else:
                test_dataset = IoTDataset(processed_test_data, test_label)
            
            train_dataset = IoTDataset(processed_train_data, train_label)
            valid_dataset = IoTDataset(processed_valid_data, valid_label)
            # dev_dataset = IoTDataset(processed_dev_data, dev_label)
            
            
            # indices = np.random.choice(processed_abnormal_data.shape[0], 3000, replace=False)
            # unique_values, counts = np.unique(abnormal_label[indices], return_counts=True)
            # print(f"Abnormal data: {unique_values} - {counts}")
            # abnormal_dataset = IoTDataset(processed_abnormal_data[indices], abnormal_label[indices])
            abnormal_dataset = IoTDataset(processed_abnormal_data, abnormal_label)
            
            test_dataset = ConcatDataset([test_dataset, abnormal_dataset])

            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                pin_memory=True
            )
            valid_loader = DataLoader(
                dataset=valid_dataset,
                batch_size=batch_size,
                pin_memory=True
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                pin_memory=True
            )
            
            # indices = np.random.choice(processed_dev_data.shape[0], 200, replace=False)
            client_info.append({
                "device": device['name'],
                "save_dir": "",
                "train_loader": train_loader,
                "valid_loader": valid_loader,
                "test_loader": test_loader,
                "test_dataset": (processed_test_data, test_label),
                "dev_normal_dataset": dev_normal_data
            })
        for update_type in ["avg", "fedprox", "mse_avg", "decentralized", "combined"]:
        # for update_type in ["fedprox"]:
        # for update_type in ["mse_avg"]:
            # for model_type in ["autoencoder"]:
            for model_type in ["hybrid", "autoencoder"]:
                for run in range(num_runs):
                    set_seeds(run*10000)
                    for client in client_info:
                        client['save_dir'] = os.path.join(f"Checkpoint/{network_size}/{no_Exp}/{run}/ClientModel", scen_name, model_type, update_type, client['device'])
                    global_worse = 0
                    min_val_loss = float("inf")
                    if True:
                        # random.seed(run*10000)
                        
                        # devices_list = config['devices_list']

                        directory = f'Checkpoint/Results/Update/{network_size}/{no_Exp}/Run_{run}/{metric}'
                        if not os.path.exists(directory):
                            os.makedirs(directory)

                        # Check if the file exists and delete its content if it does
                        filename = f'{directory}/{scen_name}_{num_participants}_{model_type}_{update_type}_results.json'
                        open(filename, 'w').close()
                        
                        if model_type == "hybrid":
                            global_model = Shrink_Autoencoder(input_dim=dim_features,
                                                                output_dim=dim_features,
                                                                shrink_lambda=shrink_lambda,
                                                                latent_dim=11,
                                                                hidden_neus=50)
                            
                            global_aggregator = GlobalAggregator(global_model, update_type=update_type)
                            
                            # Calculate the minimum length of all clients' datasets
                            min_len = min([len(client['dev_normal_dataset']) for client in client_info])

                            # Sample min_len data points from each client's dataset and create dev_dataset
                            dev_dataset = []
                            for client in client_info:
                                sample_data = client['dev_normal_dataset'].sample(n=min_len)
                                dev_dataset.append(sample_data)
                                # client['dev_normal_dataset'] = client['dev_normal_dataset'].drop(sample_data.index)

                            # Concatenate all the sampled data into a single numpy array
                            dev_dataset = np.concatenate(dev_dataset, axis=0)

                            global_aggregator.create_dev_dataset({"dataset": dev_dataset})
                            
                            # Now all clients' datasets have the same size
                            
                            # indices = np.random.choice(processed_dev_data.shape[0], 200, replace=False)
                            # dev_dataset = np.concatenate([client['dev_normal_dataset'][0] for client in client_info], axis=0)
                            # dev_label = np.concatenate([client['dev_normal_dataset'][1] for client in client_info], axis=0)
                            # global_aggregator.create_dev_dataset({"dataset": dev_dataset, "label": dev_label})
                            
                            # dev_dataset = np.concatenate([client['dev_normal_dataset'][0][indices] for client in client_info], axis=0)
                            # dev_label = np.concatenate([client['dev_normal_dataset'][1][indices] for client in client_info], axis=0)
                            # global_aggregator.create_dev_dataset({"dataset": dev_dataset, "label": dev_label})
                        
                            # global_test_data = np.concatenate([client['test_dataset'][0] for client in client_info], axis=0)
                            # global_test_label = np.concatenate([client['test_dataset'][1] for client in client_info], axis=0)
                            # global_test_dataset = IoTDataset(global_test_data, global_test_label)
                            # global_test_dataloader = DataLoader(
                            #     dataset=global_test_dataset,
                            #     batch_size=batch_size,
                            #     pin_memory=True
                            # )
                            
                            # Start training process
                            results = []
                            client_latent = {}
                            
                            # Create network topology
                            if update_type == "decentralized" or update_type == "combined":
                                if topology_type == "ring":
                                    G = nx.Graph()
                                    for i in range(len(client_info)):
                                        G.add_edge(i, (i + 1) % len(client_info))
                                elif topology_type == "random":
                                    G = nx.Graph()
                                    for i in range(len(client_info)):
                                        G.add_node(i)
                                    for i in range(len(client_info)):
                                        num_edges = random.randint(2, 4)
                                        for _ in range(num_edges):
                                            j = random.randint(0, len(client_info) - 1)
                                            if i != j:
                                                G.add_edge(i, j)
                                elif topology_type == "mesh":
                                    G = nx.complete_graph(len(client_info))
                                
                                logging.info(f"\n{'='*50}")
                                logging.info(f"Network Topology Information:")
                                logging.info(f"Type: {topology_type}")
                                logging.info(f"Number of nodes: {len(client_info)}")
                                logging.info(f"Average node degree: {sum(dict(G.degree()).values())/len(client_info):.2f}")
                                logging.info(f"Network diameter: {nx.diameter(G)}")
                                logging.info(f"Network density: {nx.density(G):.3f}")
                                logging.info(f"{'='*50}\n")

                            logging.info(f"\n{'='*50}")
                            logging.info(f"Starting Training")
                            logging.info(f"Model Type: {model_type}")
                            logging.info(f"Update Type: {update_type}")
                            logging.info(f"Number of Clients: {len(client_info)}")
                            logging.info(f"Participation Rate: {num_participants*100}%")
                            logging.info(f"{'='*50}\n")

                            # Initialize client models
                            client_models = []
                            for client in client_info:
                                if model_type == "hybrid":
                                    model = Shrink_Autoencoder(input_dim=dim_features,
                                                             output_dim=dim_features,
                                                             shrink_lambda=shrink_lambda,
                                                             latent_dim=11,
                                                             hidden_neus=50)
                                else:
                                    model = Autoencoder(input_dim=dim_features,
                                                      output_dim=dim_features,
                                                      latent_dim=11,
                                                      hidden_neus=50)
                                client_models.append(model)

                            for round in range(num_rounds):
                                client_latent[round] = {}
                                
                                if update_type == "combined":
                                    logging.info(f"\n{'='*50}")
                                    logging.info(f"Round {round+1}/{num_rounds}")
                                    logging.info(f"Using Combined Approach (FedProx + MSE + Decentralized)")
                                    
                                    # Prepare client models for combined update
                                    selected_idx = random.sample([i for i in range(len(client_info))], int(num_participants*len(client_info)))
                                    selected_clients = [client_info[i] for i in selected_idx]
                                    
                                    client_weights = []
                                    for i, client in enumerate(selected_clients):
                                        client_weights.append((
                                            copy.deepcopy(client_models[i]),
                                            client["train_loader"],
                                            client["valid_loader"]
                                        ))
                                    
                                    # Perform combined update
                                    updated_models = global_aggregator.combined_update(client_weights)
                                    
                                    # Update client models
                                    for i, model in enumerate(updated_models):
                                        client_models[selected_idx[i]] = model[0]
                                    
                                    # Use the last model as global model
                                    global_aggregator.model = client_models[-1]
                                    
                                elif update_type == "decentralized":
                                    logging.info(f"\n{'='*50}")
                                    logging.info(f"Round {round+1}/{num_rounds}")
                                    logging.info(f"Network Communication Pattern:")
                                    
                                    # Each node trains locally
                                    for i, client in enumerate(client_info):
                                        device_trainer = ClientTrainer(model=client_models[i],
                                                                     save_dir=client['save_dir'],
                                                                     epoch=epoch,
                                                                     lr_rate=lr_rate,
                                                                     update_type=update_type)
                                        device_trainer.run(client["train_loader"], client["valid_loader"])
                                    
                                    # Select aggregator through voting
                                    aggregator_idx = select_aggregator(client_models, client_info, device_trainer.device, G)
                                    logging.info(f"Selected {client_info[aggregator_idx]['device']} as aggregator for round {round+1}")
                                    
                                    # Calculate MSE scores for all models using dev_dataset
                                    mse_scores = []
                                    for i, model in enumerate(client_models):
                                        model.eval()
                                        with torch.no_grad():
                                            _, generated_data, _ = model(torch.Tensor(global_aggregator.dev_dataset).to(device_trainer.device))
                                            sim_score = torch.nn.MSELoss(reduction='mean')(torch.Tensor(global_aggregator.dev_dataset).to(device_trainer.device), generated_data)
                                            mse_scores.append(1/sim_score)  # Lower MSE = higher weight
                                    
                                    # All nodes aggregate with the voted aggregator
                                    for i in range(len(client_info)):
                                        if i == aggregator_idx:
                                            # Aggregator node aggregates with its neighbors
                                            neighbors = list(G.neighbors(i))
                                            if not neighbors:
                                                continue
                                                
                                            # Collect models from neighbors
                                            neighbor_models = [client_models[neighbor] for neighbor in neighbors]
                                            neighbor_models.append(client_models[i])  # Include own model
                                            
                                            # Calculate weights based on MSE scores
                                            neighbor_scores = [mse_scores[neighbor] for neighbor in neighbors]
                                            neighbor_scores.append(mse_scores[i])  # Include own score
                                            
                                            # Give aggregator higher weight
                                            weights = [score * 1.5 if j == i else score * 0.9 
                                                     for j, score in enumerate(neighbor_scores)]
                                            weights = [w / sum(weights) for w in weights]  # Renormalize
                                            
                                            # Aggregate models
                                            avg_weights = {}
                                            for key in neighbor_models[0].state_dict().keys():
                                                avg_weights[key] = sum(model.state_dict()[key] * weight 
                                                                     for model, weight in zip(neighbor_models, weights))
                                            
                                            # Update aggregator's model
                                            client_models[i].load_state_dict(avg_weights)
                                            logging.info(f"Aggregator {client_info[i]['device']} aggregated with neighbors: {[client_info[j]['device'] for j in neighbors]}")
                                            logging.info(f"Neighbor weights: {dict(zip([client_info[j]['device'] for j in neighbors + [i]], weights))}")
                                        else:
                                            # Non-aggregator nodes aggregate with the aggregator
                                            aggregator_model = client_models[aggregator_idx]
                                            own_model = client_models[i]
                                            
                                            # Calculate weights based on MSE scores
                                            own_score = mse_scores[i]
                                            aggregator_score = mse_scores[aggregator_idx]
                                            
                                            # Give aggregator higher weight
                                            own_weight = own_score * 0.4  # Reduce own weight
                                            aggregator_weight = aggregator_score * 1.6  # Increase aggregator weight
                                            
                                            # Normalize weights
                                            total_weight = own_weight + aggregator_weight
                                            own_weight = own_weight / total_weight
                                            aggregator_weight = aggregator_weight / total_weight
                                            
                                            # Aggregate models
                                            avg_weights = {}
                                            for key in own_model.state_dict().keys():
                                                avg_weights[key] = (own_model.state_dict()[key] * own_weight + 
                                                                  aggregator_model.state_dict()[key] * aggregator_weight)
                                            
                                            # Update node's model
                                            client_models[i].load_state_dict(avg_weights)
                                            logging.info(f"Node {client_info[i]['device']} aggregated with aggregator {client_info[aggregator_idx]['device']}")
                                            logging.info(f"Weights - Own: {own_weight:.3f}, Aggregator: {aggregator_weight:.3f}")
                                    
                                    # Use the aggregator's model for evaluation
                                    global_aggregator.model = client_models[aggregator_idx]
                                else:
                                    # Original centralized federated learning code
                                    selected_idx = random.sample([i for i in range(len(client_info))], int(num_participants*len(client_info)))
                                    selected_clients = [client_info[i] for i in selected_idx]
                                    
                                    total_training_samples = sum([len(client['train_loader'].dataset) for client in selected_clients])
                                    
                                    client_weights = []
                                    for i, client in enumerate(selected_clients):
                                        device_trainer = ClientTrainer(model=global_aggregator.model,
                                                                     save_dir=client['save_dir'],
                                                                     epoch=epoch,
                                                                     lr_rate=lr_rate,
                                                                     update_type=update_type)
                                        device_trainer.run(client["train_loader"], client["valid_loader"])
                                        client_weights.append((copy.deepcopy(device_trainer.model.state_dict()),
                                                             total_training_samples,
                                                             len(client["train_loader"].dataset)))
                                    
                                    global_aggregator.update(local_models=client_weights)

                                # Evaluation
                                evaluator = Evaluator(global_aggregator.model, metric=metric, model_type=model_type)
                                round_results = {}
                                
                                # Calculate global validation loss
                                global_aggregator.model.eval()
                                with torch.no_grad():
                                    # Use the development dataset for validation
                                    if hasattr(global_aggregator, 'dev_dataset'):
                                        _, _, val_loss = global_aggregator.model(torch.Tensor(global_aggregator.dev_dataset).to(device_trainer.device))
                                        global_aggregator.val_loss = val_loss.item()
                                    else:
                                        # If no dev dataset, use average of client validation losses
                                        val_losses = []
                                        for client in client_info:
                                            for batch in client["valid_loader"]:
                                                _, _, loss = global_aggregator.model(batch[0].to(device_trainer.device))
                                                val_losses.append(loss.item())
                                        global_aggregator.val_loss = np.mean(val_losses)
                                
                                for i, client in enumerate(client_info):
                                    auc_score, test_latent, test_label = evaluator.evaluate(client["test_loader"], client["train_loader"])
                                    round_results[client['device']] = auc_score
                                    client_latent[round][client['device']] = (test_latent, test_label)
                                
                                round_results["global_loss"] = global_aggregator.val_loss
                                if update_type != "decentralized":
                                    round_results['join_clients'] = selected_idx
                                round_results = {f'round_{round+1}': round_results}
                                
                                # Append to the JSON file
                                with open(filename, 'a') as f:
                                    f.write(json.dumps(round_results) + '\n')
                                
                                if global_aggregator.val_loss < min_val_loss:
                                    min_val_loss = global_aggregator.val_loss
                                    global_worse = 0
                                    logging.info("New best model found!")
                                
                                if global_aggregator.val_loss >= min_val_loss:
                                    global_worse += 1
                                    if global_worse > global_patience:
                                        logging.info(f"\n{'='*50}")
                                        logging.info("Early stopping triggered!")
                                        logging.info(f"Best loss: {min_val_loss:.6f}")
                                        logging.info(f"{'='*50}\n")
                                        break

                            logging.info(f"\n{'='*50}")
                            logging.info("Training Complete!")
                            logging.info(f"Final Global Loss: {global_aggregator.val_loss:.6f}")
                            logging.info(f"Best Loss: {min_val_loss:.6f}")
                            logging.info(f"{'='*50}\n")

                            # store latent data of SAE and SAE_MSEFed for all rounds
                            # Define the file path
                            file_path = f'Checkpoint/LatentData/{network_size}/{no_Exp}/Run_{run}/latent_{model_type}_{update_type}.pkl'

                            # Create the directory if it does not exist
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)

                            # Now you can safely write the file
                            with open(file_path, 'wb') as f:
                                pickle.dump(client_latent, f)
                            
                        if model_type == "autoencoder":
                            global_model = Autoencoder(input_dim=dim_features,
                                                                output_dim=dim_features,
                                                                latent_dim=11,
                                                                hidden_neus=50)
                            
                            global_aggregator = GlobalAggregator(global_model, update_type=update_type)
                            # Calculate the minimum length of all clients' datasets
                            min_len = min([len(client['dev_normal_dataset']) for client in client_info])

                            # Sample min_len data points from each client's dataset and create dev_dataset
                            dev_dataset = []
                            for client in client_info:
                                sample_data = client['dev_normal_dataset'].sample(n=min_len)
                                dev_dataset.append(sample_data)
                                # client['dev_normal_dataset'] = client['dev_normal_dataset'].drop(sample_data.index)

                            # Concatenate all the sampled data into a single numpy array
                            dev_dataset = np.concatenate(dev_dataset, axis=0)

                            global_aggregator.create_dev_dataset({"dataset": dev_dataset})
                            
                            # dev_dataset = np.concatenate([client['dev_normal_dataset'][0] for client in client_info], axis=0)
                            # dev_label = np.concatenate([client['dev_normal_dataset'][1] for client in client_info], axis=0)
                            # global_aggregator.create_dev_dataset({"dataset": dev_dataset, "label": dev_label})
                            
                            
                            
                            # global_test_data = np.concatenate([client['test_dataset'][0] for client in client_info], axis=0)
                            # global_test_label = np.concatenate([client['test_dataset'][1] for client in client_info], axis=0)
                            # global_test_dataset = IoTDataset(global_test_data, global_test_label)
                            # global_test_dataloader = DataLoader(
                            #     dataset=global_test_dataset,
                            #     batch_size=batch_size,
                            #     pin_memory=True
                            # )
                            
                            # Start training process
                            results = []
                            for round in range(num_rounds):
                                dev_dataset = []
                                dev_label = []
                                dev_dataset = []
                                dev_label = []
                                # selected_idx = random.sample([i for i in range(len(client_info))], int(num_participants*len(client_info)))
                                # selected_clients = [client_info[i] for i in selected_idx]
                                # for client in client_info:
                                #     # indices = np.random.choice(client['dev_normal_dataset'].shape[0], 50, replace=False)
                                #     n_samples = min(20, len(client['dev_normal_dataset']))
                                #     sample_data = client['dev_normal_dataset'].sample(n=n_samples)
                                #     dev_dataset.append(sample_data)
                                #     client['dev_normal_dataset'] = client['dev_normal_dataset'].drop(sample_data.index)
                                
                                # dev_dataset = np.concatenate(dev_dataset, axis=0)
                                # dev_label = np.concatenate(dev_label, axis=0)
                                # dev_dataset = np.concatenate([client['dev_normal_dataset'] for client in client_info], axis=0)
                                # global_aggregator.create_dev_dataset({"dataset": dev_dataset, "label": dev_label})
                                # global_aggregator.create_dev_dataset({"dataset": dev_dataset})
                                
                                # Choose clients to train
                                # random.seed(round*1234)
                                # num_participants = random.uniform(0,1)
                                
                                selected_idx = random.sample([i for i in range(len(client_info))], int(num_participants*len(client_info)))
                                selected_clients = [client_info[i] for i in selected_idx]
                                
                                total_training_samples = sum([len(client['train_loader'].dataset) for client in selected_clients])
                                
                                client_weights = []
                                # if round == 0:
                                for i, client in enumerate(selected_clients):
                                    logging.info("Training local model...")
                                    device_trainer = ClientTrainer(model=global_aggregator.model, \
                                        save_dir=client['save_dir'], epoch=epoch, update_type=update_type, lr_rate=lr_rate)
                                    device_trainer.run(client["train_loader"], client["valid_loader"])
                                    # client_weights.append(copy.deepcopy(device_trainer.model.state_dict()))
                                    client_weights.append((copy.deepcopy(device_trainer.model.state_dict()), total_training_samples, len(client["train_loader"].dataset)))
                                    logging.info(f"Client {i} training done!")
                                
                                logging.info(f"Round {round+1}/{num_rounds} - Updating global model")
                                
                                # client_weights = random.sample(client_weights, int(num_participants * len(client_weights)))
                                global_aggregator.update(local_models=client_weights)

                                logging.info(f"Round {round+1}/{num_rounds} - Updated global model - \
                                    Global loss: {global_aggregator.val_loss}")
                                
                                logging.info("Training done! Evaluating...")
                                # evaluate the model in clients
                            
                                evaluator = Evaluator(global_aggregator.model, metric=metric, model_type=model_type)
                                round_results = {}
                                for i, client in enumerate(client_info):
                                    logging.info(f"Evaluating client {i} - name: {client['device']}")
                                    auc_score = evaluator.evaluate(client["test_loader"], client["train_loader"])
                                    round_results[client['device']] = auc_score
                                round_results["global_loss"] = global_aggregator.val_loss
                                round_results['join_clients'] = selected_idx
                                round_results = {f'round_{round+1}': round_results}
                                
                                # Append to the JSON file
                                with open(filename, 'a') as f:
                                    f.write(json.dumps(round_results) + '\n')
                                
                                if global_aggregator.val_loss < min_val_loss:
                                    min_val_loss = global_aggregator.val_loss
                                    global_worse = 0
                                
                                if global_aggregator.val_loss >= min_val_loss:
                                    global_worse += 1
                                    if global_worse > global_patience:
                                        logging.info("Early stopping in global round!")
                                        break
                                    