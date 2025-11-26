import pandas as pd

def load_dataset():
    url = "https://raw.githubusercontent.com/jchergu/marm/main/data/dataset.csv"
    df = pd.read_csv(url)
    return df