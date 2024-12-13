import scanpy as sc
import pandas as pd


adata = sc.read("/gstore/project/crc_recursion_2/crc-NGS5368/DS000015514/raw_qc.h5ad") 

bdata = adata[adata.obs["Sample"].isin(["SAM24437819_rep1","SAM24437819_rep2"])].copy()


df1 = pd.read_csv("/gstore/scratch/u/ghaffars/scTE/APC_data/rep2/out_all.csv")
df2 = pd.read_csv("/gstore/scratch/u/ghaffars/scTE/APC_data/rep1/out_all.csv")

df1["Barcode"] = df1["barcodes"].apply(lambda x:x.split('-')[0])
df1.index = "SAM24437819_rep1"+'-'+df1["Barcode"]
df1.index.name=None
df1=df1.drop(columns="barcodes")

A= [i for i in df1.columns.tolist() if i in bdata.var.index.tolist()]

df2["Barcode"] = df2["barcodes"].apply(lambda x:x.split('-')[0])
df2.index = "SAM24437819_rep2"+'-'+df2["Barcode"]
df2.index.name=None
df2=df2.drop(columns="barcodes")

df = pd.concat([df1,df2]).drop_duplicates()
df=df.drop(columns="Barcode")
df = df[A]

merged_df = bdata.obs.join(df, how='left')
merged_df = merged_df[A].copy()

bdata.uns["scTE"]= merged_df.copy()

bdata.write("/gstore/project/crc_recursion_2/crc-NGS5368/DS000015514/raw_qc_scTE.h5ad")