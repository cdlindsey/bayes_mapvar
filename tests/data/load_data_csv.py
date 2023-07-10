''' get ground truth test data for test output results '''
import os
import pandas as pd

def load_data_csv(dataset_name):
    ''' path logic '''
    data_dir = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(data_dir, dataset_name))
