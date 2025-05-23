# FedMERL: Bridging Environmental Heterogeneity in Federated Reinforcement Learning through Meta-Experience Learning
### Problem Statement

FRL encounters significant hurdles that impede its widespread adoption and effectiveness. A primary challenge lies in the environmental heterogeneity inherent in decentralized systems. In FRL, multiple agents operate in distinct environments, each characterized by unique state transition dynamics and reward structures. This diversity leads to non-identically distributed (non-i.i.d.) data across agents, complicating the learning process.

### Motivation

To address these complementary strengths, we propose FedMERL(Federated Meta-Experience Reinforcement Learning). FedMERL integrates sharpness-aware meta-optimization to learn initialization parameters that minimize sensitivity across clients, and an experience vector module that encodes and weights high-level signals from prior training rounds. These vectors are aggregated and shared to guide local updates, enabling agents to leverage collective knowledge while maintaining personalized adaptation.
### Project Structure

FedMERL/
├── environment.yml             # Conda environment configuration
├── main/                       # Main training and coordination scripts
│   ├── fedregtd3.py            # Federated TD3 implementation
│   ├── FedMERL.py                # Main entry point
│   └── __init__.py
├── models/                     # Neural network model definitions
│   ├── Conv_model.py
│   └── Network.py
├── non_stationary_envs/       # Various non-stationary environments
│   ├── Cartpole.py
│   ├── car_dynamics.py
│   ├── ...
│   └── assets/                # Visual assets for environments
├── utils/                      # Utility functions and wrappers
│   ├── Memory.py
│   ├── Tools.py
│   └── wrappers.py
├── outputs/                    # Output directory for logs/models
└── NPYViewer-master/           # Tool for visualizing .npy sample data

### Setup & Usage
```shell
conda env create -f environment.yml

### Run the Model

```shell
# run FedMERL
python FedMERL.py


