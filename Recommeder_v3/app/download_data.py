"""
Downloads the real UCI Online Retail dataset from its GitHub mirror.

Original source: UCI Machine Learning Repository
  https://archive.ics.uci.edu/dataset/352/online+retail
  Citation: Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5BW33 — Licensed CC BY 4.0
"""
import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/eaintkyawthmu/UCI_Online_Retail_Dataset_Cleaned_Version/master/Cleaned_UCI_Online_Sale_Dataset.csv"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_PATH = os.path.join(DATA_DIR, "online_retail.csv")


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_PATH):
        print(f"Already present at {DATA_PATH}")
        return DATA_PATH
    print(f"Downloading real Online Retail dataset from {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print(f"Saved to {DATA_PATH}")
    return DATA_PATH


if __name__ == "__main__":
    download()
