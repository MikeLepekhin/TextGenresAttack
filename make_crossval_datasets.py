import numpy as np
import pandas as pd
import sys


def split_dataset_crossval(dataset_filename):
    print("Going to split " + dataset_filename)
    test_df = pd.read_csv(dataset_filename).sample(frac=1, random_state=42)
    n = len(test_df)
    
    for val_step in range(5):
        lb = int(0.2 * val_step * n)
        rb = int(0.2 * (val_step+1) * n)
        test_df.iloc[lb:rb, :].to_csv(dataset_filename + '_crossval_' + str(val_step))
        
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("the script takes exactly one argument - the path to the csv file that you want to split")
        return

    split_dataset_crossval(sys.argv[1])
