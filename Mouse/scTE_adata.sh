#!/bin/bash

#SBATCH -p himem
#SBATCH --mem=512G
#SBATCH --qos=short
#SBATCH -c 8

ml spaces/gpy
ml gpy_gpu/Python310

python ./scTE_adata.py --dataset_home "/gstore/data/dld1_concerto/scTE/NGS5400_MouseAKPS"