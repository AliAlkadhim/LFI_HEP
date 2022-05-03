#!/bin/bash

export TRAINING_DATA='data/two_parameters_N_M_Uniformly_sampled_1M.'
export RUN_NAME='UNIFORM_5K_CDF'

# python src/Generate_Training_Data_One_param.py --D 1 --Bprime 10000

# python src/Generate_Training_Data_One_param.py --D 1 --Bprime 10000
# python src/Run_Regressor_Training.py --D 1
python src/Inference_two_params.py 
