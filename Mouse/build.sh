#!/bin/bash


# SBATCH --qos=short
# SBATCH -c 8


ml Python/3.9.6-GCCcore-11.2.0/


scTE_build -g mm10 # Mouse

