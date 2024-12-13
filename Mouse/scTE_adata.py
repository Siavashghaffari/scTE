import scanpy as sc
import pandas as pd
import anndata as ad
import argparse
import pickle
import os
from scipy.sparse import csr_matrix

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process paths for scTE data.")
    parser.add_argument(
        "--dataset_home", 
        type=str, 
        required=True, 
        help="Path to the dataset home directory"
    )
    args = parser.parse_args()
    
    # Assign Dataset_Home from the parsed arguments
    Dataset_Home = args.dataset_home
    
    # Define file paths
    tename_path = os.path.join(Dataset_Home, f'tename.pkl')
    adata_path = os.path.join(Dataset_Home, f'raw_qc.h5ad')
    csv_path = os.path.join(Dataset_Home, f'scTE.csv')
    adata_scTE_path = os.path.join(Dataset_Home, f'raw_qc_scTE.h5ad')


    ## Read adata and filtering
    adata = sc.read(adata_path) 
    bdata = adata[adata.obs["Sample"].isin(['SAM24438240_rep1', 'SAM24438240_rep2', 'SAM24438240_rep3'])].copy()


    df1 = pd.read_csv("/gstore/scratch/u/ghaffars/scTE/APKS_data/AKPS/1/out.csv")
    df2 = pd.read_csv("/gstore/scratch/u/ghaffars/scTE/APKS_data/AKPS/2/out.csv")
    df3 = pd.read_csv("/gstore/scratch/u/ghaffars/scTE/APKS_data/AKPS/3/out.csv")


    df1["Barcode"] = df1["barcodes"].apply(lambda x:x.split('-')[0])
    df1.index = "SAM24438240_rep1"+'-'+df1["Barcode"]
    df1.index.name=None
    df1 = df1.drop(columns="barcodes")

    df2["Barcode"] = df2["barcodes"].apply(lambda x:x.split('-')[0])
    df2.index = "SAM24438240_rep2"+'-'+df2["Barcode"]
    df2.index.name=None
    df2 = df2.drop(columns="barcodes")

    df3["Barcode"] = df3["barcodes"].apply(lambda x:x.split('-')[0])
    df3.index = "SAM24438240_rep3"+'-'+df3["Barcode"]
    df3.index.name=None
    df3 = df3.drop(columns="barcodes")

    df = pd.concat([df1,df2,df3]).drop_duplicates()
    df = df.drop(columns="Barcode")

    ## keep only qc passed cells
    df_qc = bdata.obs.iloc[:,0:0]
    df = df_qc.join(df, how='left')

    ## Write down csv file of csv info
    df.to_csv(csv_path)


    with open(tename_path, "rb") as fp:   #Pickling
        tename_list = pickle.load(fp)
        
    ## Write down anndata
    cdata = ad.AnnData(csr_matrix(df.values))
    cdata.obs_names = df.index
    cdata.var_names = df.columns
    tename = [True if k in tename_list else False for k in list(df.columns)]
    cdata.var["is_te"]= tename
    cdata.write(adata_scTE_path)
    
if __name__ == "__main__":
    main()



#df = df.reindex(columns=bdata.var.index.tolist()).copy()
#merged_df = bdata.obs.join(df, how='left')
#merged_df = merged_df[bdata.var.index.tolist()].copy()
#array = merged_df.to_numpy()
#bdata.layers['scTE'] = array.copy()
#bdata.write("/gstore/project/crc_recursion_2/NGS5400/AKPS/DS000015552/raw_qc_scTE.h5ad")
