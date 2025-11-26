import pandas as pd

def load_dataset():
    url = "https://raw.githubusercontent.com/<user>/<repo>/main/data/dataset.csv"
    df = pd.read_csv(url)
    return df