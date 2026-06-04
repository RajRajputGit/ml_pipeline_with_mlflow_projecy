import pandas as pd
import sys
import yaml
import os

# loading the params of yaml file
params = yaml.safe_load(open("params.yaml","rb"))["preprocess"]

def preprocess_data(input_path, output_path):
    # read the data
    df = pd.read_csv(input_path)
    
    # drop the null values
    df.dropna(inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # save the preprocessed data
    df.to_csv(output_path, index=False)

    print(f"Preprocessed data saved at{output_path}")

if __name__ == "__main__":
    preprocess_data(params["input"],params["output"])