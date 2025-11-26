from dataset_loader import load_dataset

df = load_dataset()

def check_consistency(df):
    print(df.head())
    print(df.info())
    print(df.describe())

check_consistency(df)
