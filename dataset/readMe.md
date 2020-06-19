**This folder contains all code used to generate the Proprioceptive Character Recognition Task dataset.**  

* To explore, please start here - `Figure2_DatasetGeneration` notebook contains a walk-through of the entire data generation process, starting from pen-tip trajectories up to spindle firing rates.
* `uci_tracjectorydata.py` contains code to reformat the [ dataset by Williams et al.](https://archive.ics.uci.edu/ml/datasets/Character+Trajectories) into a suitable format for our processing. References are contained in this website and our paper.
* `generate_startpoints.py` was used to compute the reachable starting points in the workspace of a simple 2-link arm model, defined in `pcr_data_utils.py`.  
* `generate_data.py` is a script that can be used to generate datapoints of a given character label in a given plane (horizontal/vertical).

In order to run `Figure2_DatasetGeneration`, please make sure that you use the provided docker container, since it has opensim python API installed on it.
