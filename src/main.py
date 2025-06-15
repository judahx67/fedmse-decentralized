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
# from Trainer import GlobalAggregator
from Evaluator import Evaluator

import logging
import torch.nn as nn

# Configure the logging module
logging.basicConfig(level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(levelname)s - %(message)s')


num_participants = 0.5 # 0.5
epoch = 5 #5 #100
num_rounds = 3 #5 #20
lr_rate = 1e-3
shrink_lambda = 1 #5 #10
network_size = 10 #50
data_seed = 1234
no_Exp = f"nonIID_Exp122_Rerun_{epoch}epoch_10client_lr0001_lamda{shrink_lambda}_ratio{num_participants*100}"
checkpoint_dir = f"Checkpoint/Results/Update/{network_size}/{no_Exp}"
# Verification method selector
verification_method = "dev"  # Options: "dev" or "val" - use development dataset or validation data for verification
#no_Exp = f"IID-Update_Exp6_scale_{epoch}epoch_{network_size}client_{num_rounds}rounds_lr{lr_rate}_lamda{shrink_lambda}_ratio{num_participants*100}_dataseed{data_seed}"

num_runs = 1 #5
batch_size = 12

new_device = True
min_val_loss = float("inf")
global_patience = 1
global_worse = 0
metric = "AUC" #AUC or classification
# Define all combinations to try
model_types = ["hybrid", "autoencoder"]
update_types = ["avg", "fedprox", "mse_avg"]
dim_features = 115   #nba-iot: 115; cic-2023: 46

scen_name = 'FL-IoT' 

#config_file = f"Configuration/scen2-nba-iot-50clients.json"
#config_file = "Configuration/scen2-nba-iot-10clients_noniid.json"
config_file = "Configuration/scen2-nba-iot-10clients.json"
#config_file = "Configuration/kitsune-iot-10clients.json"
# config_file = "Configuration/cic-config.json"

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # Calculate total combinations for progress tracking
    total_combinations = len(update_types) * len(model_types) * num_runs
    current_combination = 0
    
    # Dictionary to store best metrics for summary
    best_metrics = {mt: {ut: float('-inf') for ut in update_types} for mt in model_types}
    
    # Print initial parameters
    logging.info("\n" + "="*50)
    logging.info("Training Parameters:")
    logging.info("="*50)
    logging.info(f"Number of runs: {num_runs}")
    logging.info(f"Number of rounds per run: {num_rounds}")
    logging.info(f"Epochs per round: {epoch}")
    logging.info(f"Learning rate: {lr_rate}")
    logging.info(f"Shrink lambda: {shrink_lambda}")
    logging.info(f"Network size: {network_size}")
    logging.info(f"Number of participants ratio: {num_participants}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Data seed: {data_seed}")
    logging.info(f"Experiment name: {no_Exp}")
    logging.info(f"Model types: {model_types}")
    logging.info(f"Update types: {update_types}")
    logging.info(f"Total combinations to run: {total_combinations}")
    logging.info("="*50 + "\n")

    # Loop over all combinations
    for model_type in model_types:
        for update_type in update_types:
            for run in range(num_runs):
                current_combination += 1
                logging.info(f"\nStarting combination {current_combination}/{total_combinations}")
                logging.info(f"Model type: {model_type}, Update type: {update_type}, Run: {run + 1}/{num_runs}")
                
                set_seeds(run * 10000)  # Set different seed for each run
                random.seed(data_seed)
                np.random.seed(data_seed)
                
                try:
                    logging.info("Loading configuration...")
                    with open(config_file, "r") as config_file:
                        config = json.load(config_file)
                except Exception as e:
                    logging.info("Failed to load configuration.")
                
                devices_list = random.sample(config['devices_list'], network_size)
                client_info = []
                
                # Initialize clients
                clients = []
                for device in devices_list:
                    logging.info("Creating metadata for client...")
                    normal_data_path = os.path.join(config['data_path'], device["normal_data_path"])
                    abnormal_data_path = os.path.join(config['data_path'], device["abnormal_data_path"])
                    test_new_normal_data_path = os.path.join(config['data_path'], device["test_normal_data_path"])
                    
                    logging.info("Loading data from {}...".format(device['name']))
                    
                    normal_data = load_data(normal_data_path)
                    normal_data = normal_data.sample(frac=1).reset_index(drop=True)
                    abnormal_data = load_data(abnormal_data_path)
                    abnormal_data = abnormal_data.sample(frac=1).reset_index(drop=True)
                    
                    if new_device:
                        new_normal_data = load_data(test_new_normal_data_path)
                    
                    device_name = device['name']
                    print(f"{device_name} has {len(normal_data)} normal data and {len(abnormal_data)} abnormal data")
                    
                    # Split data
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
                    
                    client_info.append({
                        "device": device_name,
                        "train_loader": train_loader,
                        "valid_loader": valid_loader,
                        "test_loader": test_loader,
                        "test_dataset": (processed_test_data, test_label),
                        "dev_normal_dataset": dev_normal_data,
                        "save_dir": os.path.join(f"Checkpoint/{network_size}/{no_Exp}/{run}/ClientModel", 
                                            scen_name, model_type, update_type, device_name)
                    })

                # Initialize clients with P2P
                trainers = []
                client_ids = list(range(len(client_info)))
                
                # Create development dataset from all clients' dev data
                min_len = min([len(client['dev_normal_dataset']) for client in client_info])
                dev_dataset = []
                for client in client_info:
                    sample_data = client['dev_normal_dataset'].sample(n=min_len)
                    dev_dataset.append(sample_data)
                dev_dataset = pd.concat(dev_dataset, axis=0)
                
                # Process development dataset
                data_processor = IoTDataProccessor(scaler="standard")
                processed_dev_data, _ = data_processor.fit_transform(dev_dataset)
                
                for i, client in enumerate(client_info):
                    client['save_dir'] = os.path.join(f"Checkpoint/{network_size}/{no_Exp}/{run}/ClientModel", 
                                            scen_name, model_type, update_type, client['device'])
                    
                    # Create client trainer
                    if model_type == "hybrid":
                        model = Shrink_Autoencoder(input_dim=dim_features,
                                                 output_dim=dim_features,
                                                 shrink_lambda=shrink_lambda)
                    else:
                        model = Autoencoder(input_dim=dim_features,
                                          output_dim=dim_features)
                    
                    trainer = ClientTrainer(
                        model=model,
                        loss_function=nn.MSELoss,
                        optimizer=torch.optim.Adam,
                        epoch=epoch,
                        batch_size=batch_size,
                        lr_rate=lr_rate,
                        update_type=update_type,
                        patience=global_patience,
                        save_dir=client['save_dir'],
                        client_id=i,
                        model_type=model_type,
                        verification_method=verification_method,
                        verification_threshold=3.0,
                        performance_threshold=0.002
                    )
                    
                    # Initialize development dataset for aggregation
                    trainer.create_dev_dataset({"dataset": processed_dev_data})
                    trainers.append(trainer)

                # Connect clients in a peer-to-peer network
                for i, client in enumerate(trainers):
                    # Connect to all other clients except self
                    peers = [c for j, c in enumerate(trainers) if j != i]
                    client.connect_to_peers(peers)
                    client.validation_data = torch.Tensor(processed_valid_data)

                # Training loop
                for round in range(num_rounds):
                    logging.info(f"Starting round {round + 1}/{num_rounds}")
                    
                    # Train clients locally
                    for i, client in enumerate(trainers):
                        logging.info(f"Training client {i+1}...")
                        client.run(client_info[i]["train_loader"], client_info[i]["valid_loader"])
                        logging.info(f"Client {i+1} training done!")
                    
                    # Peer-to-peer model updates
                    logging.info("Starting peer-to-peer model updates...")
                    
                    # Each client broadcasts its model to peers
                    for client in trainers:
                        client.broadcast_model()
                    
                    # Each client updates its model based on received peer updates
                    verification_results = []
                    for i, client in enumerate(trainers):
                        client.update_from_peers()
                        verification_results.append({
                            'client_id': i,
                            'rejected_updates': client.rejected_updates,
                            'is_verified': client.rejected_updates == 0
                        })
                    
                    # Log verification results
                    logging.info("Verification results for this round:")
                    for result in verification_results:
                        logging.info(f"Client {result['client_id']}: {'Verified' if result['is_verified'] else 'Rejected'} "
                                    f"(Rejected updates: {result['rejected_updates']})")
                    
                    # Save verification results
                    verification_file = f'{checkpoint_dir}/Run_{run}/verification_results.json'
                    os.makedirs(os.path.dirname(verification_file), exist_ok=True)  # Create directory if it doesn't exist
                    with open(verification_file, 'a') as f:
                        json.dump({
                            'round': round + 1,
                            'verification_results': verification_results
                        }, f)
                        f.write('\n')

                    # Calculate metrics for each client model
                    logging.info("Calculating metrics for all models...")
                    client_metrics = []
                    for i, client in enumerate(trainers):
                        client_evaluator = Evaluator(client.model, model_type=model_type, metric=metric)
                        client_metric = client_evaluator.evaluate(client_info[i]["test_loader"], client_info[i]["train_loader"])
                        if isinstance(client_metric, tuple):
                            client_metric = client_metric[0]
                        client_metrics.append(client_metric)
                        logging.info(f"Client {i+1} {metric} score: {client_metric}")
                    
                    # Save results
                    directory = f'Checkpoint/Results/Update/{network_size}/{no_Exp}/Run_{run}/{metric}'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    
                    filename = f'{directory}/{scen_name}_{num_participants}_{model_type}_{update_type}_results.json'
                    with open(filename, 'a') as f:
                        json.dump({
                            'round': round + 1,
                            'client_metrics': [float(score) for score in client_metrics],
                            'update_type': update_type,
                            'model_type': model_type,
                            'global_loss': min(client_metrics) if client_metrics else float('inf')
                        }, f)
                        f.write('\n')
                    
                    # Check early stopping
                    if min(client_metrics) < min_val_loss:
                        min_val_loss = min(client_metrics)
                        global_worse = 0
                    else:
                        global_worse += 1
                        if global_worse > global_patience:
                            logging.info("Early stopping in global round!")
                            break
                
                # After training loop, update best metrics
                for i, trainer in enumerate(trainers):
                    client_evaluator = Evaluator(trainer.model, model_type=model_type, metric=metric)
                    client_metric = client_evaluator.evaluate(client_info[i]["test_loader"], 
                                                            client_info[i]["train_loader"])
                    if isinstance(client_metric, tuple):
                        client_metric = client_metric[0]
                    best_metrics[model_type][update_type] = max(best_metrics[model_type][update_type], client_metric)

                # Clean up GPU memory after each combination
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logging.info("Cleaned up GPU memory")

    # Print training summary
    logging.info("\nTraining Summary:")
    logging.info("="*50)
    for model_type in model_types:
        for update_type in update_types:
            logging.info(f"{model_type} + {update_type}: Best {metric} = {best_metrics[model_type][update_type]:.4f}")
    logging.info("="*50)

    # Save summary to file
    summary_file = f'Checkpoint/Results/Update/{network_size}/{no_Exp}/training_summary.json'
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump({
            'best_metrics': best_metrics,
            'metric_type': metric,
            'num_runs': num_runs,
            'network_size': network_size,
            'experiment_name': no_Exp
        }, f, indent=4)
    logging.info(f"Saved training summary to {summary_file}")
        