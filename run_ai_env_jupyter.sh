#!/bin/bash

# Activate the environment and execute the commands within a subshell
(
    eval "$(conda shell.bash hook)"

    # Load and run packages
    module load ai_training/2023.07
    #module load cudnn/8.8.1-cuda11.8.0
    jupyter lab --no-browser --ip="$(hostname)".ibex.kaust.edu.sa
    
)
