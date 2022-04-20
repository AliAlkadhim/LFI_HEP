#!/bin/bash

export TRAINING_DATA='data/TRAIN_DATA_1_PARAM_D'
export RUN_NAME='UNIFORM_100K'

##python src/Generate_Training_Data_One_param.py --D 1
python src/Run_Regressor_Training.py
python src/Inference.py
