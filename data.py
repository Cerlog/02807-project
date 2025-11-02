import gdown
from pathlib import Path
from loguru import logger

def download_dataset(url, raw_data_path):
    raw_data_path = Path(raw_data_path)
    raw_data_path.mkdir(parents=True, exist_ok=True)

    if any(raw_data_path.iterdir()):
        logger.success(f"✅ Using existing data in '{raw_data_path}'")
        return

    file_path = raw_data_path / "dataset.json"

    logger.info("Downloading dataset from Google Drive...")
    gdown.download(url, str(file_path), quiet=False, fuzzy=True, use_cookies=True)

    logger.success(f"✅ Downloaded JSON to '{file_path}'")

if __name__ == "__main__":
    # Use the direct download format — note the uc?id=... part
    DATASET_URL = "https://drive.google.com/uc?id=163KgvKNYPTV5_fArvlBiNkqb6rCDkk-Z"
    RAW_DATA_PATH = Path("./data/raw")

    download_dataset(DATASET_URL, RAW_DATA_PATH)
