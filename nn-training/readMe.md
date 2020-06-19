**This folder contains code to train neural network models of proprioception.**

* `train_convnets.py` and `train_convnets_seeded.py` contains code to train spatial-temporal and spatiotemporal neural network models on the PCRT dataset. 
* `get-weights.sh` script can be used to download the data. 
    * After extracting the data: `experiment_1` contains weights for 100 networks, all trained on the PCRT dataset. `experiment_4` contains 5 instantiations (initialized and trained) of the best networks of each type (spatial-temporal and spatiotemporal).
* `ConvSearchResults` and `ConvSearchResults-AnalyzedModels` contain code to gather performance numbers from all networks. 

In order to train networks, please make sure that you use the provided docker container and that your host system has CUDA and CudNN specifications met.
