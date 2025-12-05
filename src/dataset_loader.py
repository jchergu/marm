import pandas as pd

def load_dataset():
    print("[load dataset] starting...")
    url = "https://raw.githubusercontent.com/jchergu/marm/main/data/dataset.csv"
    df = pd.read_csv(url)
    print(f"[load dataset] done: {df.shape}")
    return df