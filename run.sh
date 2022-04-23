#!/bin/bash

export TRAINING_DATA='data/TRAIN_DATA_1_PARAM_D'
export RUN_NAME='UNIFORM_5K_CDF'

# python src/Generate_Training_Data_One_param.py --D 1 --Bprime 10000
python src/Run_Regressor_Training.py --D 1
python src/Inference.py --D 1
