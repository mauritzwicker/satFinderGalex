import os
import pandas as pd

def load_data(pthData, dataName):
    df_loaded = pd.read_parquet(os.path.join(pthData, dataName))
    print('Loaded {0}'.format(dataName))
    return(df_loaded)