# Fed-IoTIDS
FedMSE: Semi-supservised federated learning for IoT network intrusion detection

# Project structure

1. Data/ folder: Store data for experiments
2. Notebook/: Jupyter notebook for prepare data for training and testing
3. src/: Source code for experiments
    - Checkpoint/ --> Store the experimental report in json format and store latent data of SAE in each run
    - Configuration/ --> Store the config file to load the data
    - DataLoader/ --> Module for data preparation and preprocessing
    - Evaluator/ --> Testing phase module
    - Model/ --> Machine learning model definition
    - Trainer/ --> Modules for local training and global aggregation
    - Utils/ --> Utils for calculate the sim score
    - Visualization/ --> Jupyter notebook for visualizing the data and results