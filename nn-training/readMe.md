**This folder contains code to train neural network models of proprioception.**

* `train_convnets.py` and `train_convnets_seeded.py` contains code to train spatial-temporal and spatiotemporal neural network models on the PCRT dataset. 
* `get-weights.sh` script can be used to download the data. 
    * After extracting the data: `experiment_1` contains weights for 100 networks, all trained on the PCRT dataset. `experiment_4` contains 5 instantiations (initialized and trained) of the best networks of each type (spatial-temporal and spatiotemporal). In order to load the models and generate "neural responses", please see `code/nn-models.py`, `code/kinematics_decoding.py` and `single_cell`.
* `ConvSearchResults` and `ConvSearchResults-AnalyzedModels` contain code to gather performance numbers from all networks. 

In order to train networks, please make sure that you use the provided docker container and that your host system has CUDA and CudNN specifications met.

Each folder in `experiment_1` and `experiment_4` contain configurations and weights corresponding to a single network. To look at the configurations (hyperparameter settings) of the network, browse the `config.yaml` file inside each sub-folder. To load the model, use `load_model` from `kinematics_decoding.py`.
