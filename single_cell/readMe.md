**These folders contain the code for the analysis pipeline to extract and analyze the hidden layer activations.**  

The pipeline is run from the controls_main.py script in the command line, e.g.
```
python3 controls_main.py
```
with the following command line options:

 --S [True / False]: Compute results for Spatial-Temporal model
 --ST [True / False]: Compute results for Spatiotemporal model
 --data [True / False]: Save hidden layer activations
 --results [True / False]: Fit tuning curves
 --analysis [True / False]: Analyze strength of fits

In order to simplify reproducing results, the pipeline uses the model weights saved in `analysis-data/models` (in the shared Dropbox folder).

The local location of the `analysis-data/` folder needs to be specified as the `basefolder` variable at the beginning of the controls_main.py script.

The results are then saved in the corresponding experimental folder located in `analysis-data/`. The pipeline dynamically checks which parts of the analysis have already been completed and skips these on a new call from the command line. In order to repeat an analysis that has already been completed, either the if-flag in the main script needs to be set to `if(True)` (option exists as a comment in most places), or the corresponding folder needs to be deleted manually.
