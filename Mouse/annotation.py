import pickle
import pandas as pd

idx_path = '/gstore/data/dld1_concerto/scTE/mm10_rmsk.bed'
tename_path = '/gstore/data/dld1_concerto/scTE/NGS5400_MouseAKPS/tename.pkl'

bed = pd.read_csv(idx_path, sep="\t", header=None)

bed["TE"] = (bed[3].str.split("|").str[3]).str.split(":").str[0]
tename = bed["TE"].dropna().unique().tolist()

with open(tename_path, "wb") as fp:   #Pickling
        pickle.dump(tename, fp)