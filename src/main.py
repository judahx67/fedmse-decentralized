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


num_participants = 0.5
epoch = 1 #5 #100
num_rounds = 1 #5 #20
lr_rate = 1e-3
shrink_lambda = 1 #5 #10
network_size = 10 #50
data_seed = 1234
no_Exp = f"nonIID_Exp1_Rerun_{epoch}epoch_10client_lr0001_lamda{shrink_lambda}_ratio{num_participants*100}"
#no_Exp = f"IID-Update_Exp6_scale_{epoch}epoch_{network_size}client_{num_rounds}rounds_lr{lr_rate}_lamda{shrink_lambda}_ratio{num_participants*100}_dataseed{data_seed}"

num_runs = 1 #5
batch_size = 12

new_device = True
min_val_loss = float("inf")
global_patience = 1
global_worse = 0
metric = "AUC" #AUC or classification
model_type = "hybrid"   #autoencoder; hybrid;
update_type = "mse_avg"  #avg; fusion_avg; mse_avg
dim_features = 115   #nba-iot: 115; cic-2023: 46

scen_name = 'FL-IoT' 

#config_file = f"Configuration/scen2-nba-iot-50clients.json"
config_file = "Configuration/scen2-nba-iot-10clients_noniid.json"
# config_file = "Configuration/cic-config.json"

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    for run in range(num_runs):
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
            
            # Create client trainer
            if model_type == "hybrid":
                model = Shrink_Autoencoder(input_dim=dim_features,
                                         output_dim=dim_features,
                                         shrink_lambda=shrink_lambda)
            else:
                model = Autoencoder(input_dim=dim_features,
                                  output_dim=dim_features)
            
            client = ClientTrainer(
                model=model,
                loss_function=nn.MSELoss,
                optimizer=torch.optim.Adam,
                epoch=epoch,
                batch_size=batch_size,
                lr_rate=lr_rate,
                update_type=update_type,
                patience=global_patience,
                save_dir=os.path.join(f"Checkpoint/{network_size}/{no_Exp}/{run}/ClientModel", 
                                    scen_name, model_type, update_type, device_name)
            )
            clients.append(client)
            
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

        # Connect clients in a peer-to-peer network
        for i, client in enumerate(clients):
            # Connect to all other clients except self
            peers = [c for j, c in enumerate(clients) if j != i]
            client.connect_to_peers(peers)
            client.validation_data = torch.Tensor(processed_valid_data)

        # Training loop
        for round in range(num_rounds):
            logging.info(f"Starting round {round + 1}/{num_rounds}")
            
            # Train clients locally
            for i, client in enumerate(clients):
                logging.info(f"Training client {i+1}...")
                client.run(client_info[i]["train_loader"], client_info[i]["valid_loader"])
                logging.info(f"Client {i+1} training done!")
            
            # Peer-to-peer model updates
            logging.info("Starting peer-to-peer model updates...")
            
            # Each client broadcasts its model to peers
            for client in clients:
                client.broadcast_model()
            
            # Each client updates its model based on received peer updates
            for client in clients:
                client.update_from_peers()
            
            # Calculate AUC for each client model
            logging.info("Calculating AUC scores for all models...")
            client_auc_scores = []
            for i, client in enumerate(clients):
                client_evaluator = Evaluator(client.model, model_type=model_type, metric="AUC")
                client_auc = client_evaluator.evaluate(client_info[i]["test_loader"], client_info[i]["train_loader"])
                if isinstance(client_auc, tuple):
                    client_auc = client_auc[0]
                client_auc_scores.append(client_auc)
                logging.info(f"Client {i+1} AUC score: {client_auc}")
            
            # Save results
            directory = f'Checkpoint/Results/Update/{network_size}/{no_Exp}/Run_{run}/{metric}'
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            filename = f'{directory}/{scen_name}_{num_participants}_{model_type}_{update_type}_results.json'
            with open(filename, 'a') as f:
                json.dump({
                    'round': round + 1,
                    'client_auc_scores': [float(score) for score in client_auc_scores]
                }, f)
                f.write('\n')
        