import os
from kaggle.api.kaggle_api_extended import KaggleApi

# 自動化 Kaggle API 認證
def download_kaggle_dataset(dataset, path='wikiart'):
    if not os.path.exists(path):
        os.makedirs(path)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=path, unzip=True)

# 下載 wikiart 資料集
download_kaggle_dataset('prafulla-patel/wikiart-dataset')