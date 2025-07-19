
Peer-to-peer decentralized learning approach to the paper: [*FedMSE: Semi-supservised federated learning for IoT network intrusion detection*](https://doi.org/10.1016/j.cose.2025.104337)

This repository includes the prepared data for all subnetworks in the IoT network and the source code for reproducing the experimental results
# Project structure

1. Data/ folder: Store data for experiments
2. Notebook/: Jupyter notebook for prepare data for training and testing. You can download the original dataset from [the official website here](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot)

3. src/: Source code for experiments
    - Checkpoint/ --> Store the experimental report in json format and store latent data of SAE in each run
    - Configuration/ --> Store the config file to load the data
    - DataLoader/ --> Module for data preparation and preprocessing
    - Evaluator/ --> Testing phase module
    - Model/ --> Machine learning model definition
    - Trainer/ --> Modules for local training and global aggregation
    - Utils/ --> Utils for calculate the sim score
    - Visualization/ --> Jupyter notebook for visualizing the data and results

# How to run the experiments
1. Configure the data for experimental

```
config_file = f"Configuration/scen2-nba-iot-10clients.json"
```

2. Configure some hyper-parameters for experiment

```
num_participants = 0.5
epoch = 100
num_rounds = 20
lr_rate = 1e-5
shrink_lambda = 10
```

3. Install dependency libraries

```
pip install -r requirements.txt
```

4. Run the experiment

```
python main.py
```

# Citation


```
Nguyen, V.T., Beuran, R. (2025). FedMSE: Semi-supervised federated learning approach for IoT network intrusion detection. Computers & Security, 151, 104337. 
https://doi.org/10.1016/j.cose.2025.104337
```

}
```

