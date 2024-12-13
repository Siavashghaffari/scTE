#!/bin/bash

#SBATCH -p himem
#SBATCH --mem=512G
#SBATCH -c 8

# Define parameters
INPUT_BAM="output.bam"  # Input BAM file
THREADS=12              # Number of threads
OUTPUT_FILE="out"    # Output directory
INDEX="mm10.exclusive.idx"  # Genome index
EXPECTED_CELLS=25000    # Expected number of cells,Please set it to some value greater than avaialbale cells in the dataset
HDF5="False"            # HDF5 output (True/False)
CELL_BARCODE="CB"       # Cell barcode tag
UMI_BARCODE="UB"        # UMI barcode tag

 
# Load required modules
ml Python/3.9.6-GCCcore-11.2.0/
ml samtools/1.18-GCC-6.3.0-2.28
export PATH="${PATH}:$HOME/.local/bin"

 

# Run scTE with parameters
scTE -i "${INPUT_BAM}" \
     -p "${THREADS}" \
     -o "${OUTPUT_FILE}" \
     -x "${INDEX}" \
     --expect-cells "${EXPECTED_CELLS}" \
     --hdf5 "${HDF5}" \
     -CB "${CELL_BARCODE}" \
     -UMI "${UMI_BARCODE}"


