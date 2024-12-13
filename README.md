# scTE Pipeline
 

This repository provides a pipeline for using the **scTE** library to process and analyze single-cell transcriptomic data. The pipeline supports two datasets:

1. Human APC dataset
2. Mouse AKPS dataset 

---

## Installation

To install the **scTE** library, follow these steps: Clone the scTE repository:

```bash 
$ git clone https://github.com/JiekaiLab/scTE.git
$ cd scTE
$ python setup.py install
```

## Usage
 
### 1. Build Genome Indices

Run the `build.sh` script to build genome indices required for the scTE pipeline. 

```bash
sbatch build.sh
```

### 2. Download BAM Files

Download the BAM files for your datasets. Ensure these files are saved in an appropriate location for further processing.


### 3. Clean BAM Files

Run the `clean_bam.sh` script to clean and preprocess the BAM files, ensuring compatibility with the scTE pipeline.

```bash
sbatch clean_bam.sh
```

### 4. Run scTE Algorithm

Use the `run.sh` script to execute the scTE algorithm on the cleaned BAM files. This step will generate raw counts annotated with transposable elements (TEs). 
```bash 
 sbatch run.sh
 ```

### 5. Prepare Results in AnnData Format 

Run the `scTE_adata.sh` script to convert the scTE results into both an AnnData object and a CSV file.

- In the AnnData object, the output will be stored in `adata.X` as a sparse matrix for efficient storage and computation. 
- The CSV file, `scTE.csv`, contains column-based scTE results, which can be integrated into `adata.obs` in the next step for visualization purposes. 

```bash 
sbatch scTE_adata.sh
```

### 6. Visualize Results

Launch the `post_scTE.ipynb` Jupyter Notebook to explore and visualize the processed data. You can load the `scTE.csv` to add it to `adata.obs`. You can generate UMAP embeddings and perform additional analyses. 

## Authors and acknowledgment
This work was developed by Siavash Ghaffari. For any questions, feedback, or additional information, please feel free to reach out. Your input is highly valued and will help improve and refine this pipeline further.
