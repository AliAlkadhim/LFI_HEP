# LFI_HEP

# Likelihood-free Inference for Confidence Interval Estimation in Particle Physics

# Current Usage:
The pipeline bash script is `run.sh`, which by default runs the entire chain. For a given run, one can just specify the `RUN_NAME` in `run.sh`. Currently the chain is composed of 3 files:

1. `src/Generate_Training_Data_One_param.py`, with the arguments `--D` and `--Bprime`. These parameters can be specified in the `run.sh` script (or by running the script on its own). This file generates a file in `data` directory, with the naming scheme `<RUN_NAME>_D_eq_ <D>.csv'. 
2. `src/Run_Regressor_Training.py`, which trains the regressor using the data generated in the previous step, and saves the trained model in `models` directory.
3. `src/Inference.py`, which loads the trained regressor from the previous step and evualtes its output for new input (new $\theta$) to return $\hat{p}(D;\theta)$ and compares it to the exact calculate $p$-value.
