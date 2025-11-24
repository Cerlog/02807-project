import gdown
import subprocess
from pathlib import Path
from loguru import logger

def clean_review_json(file_path):
    """Cleaning the 'review.json' as it has some formatting issues, i.e, not closed {} brackets."""
    file_path = Path(file_path)
    if file_path.name != "review.json":
        logger.warning("Can't find 'review.json' to clean.")
        return 
    
    clean_path = file_path.with_name("review_clean.ndjson")
    
    logger.info("Cleaning 'review.json'...")
    
    with open(clean_path, "w") as out: 
            subprocess.run(
                [
                    "jq",
                    "-cR",
                    "fromjson? | select(type==\"object\")",
                    str(file_path),
                ],
                stdout=out,
                check=True,
            )
    logger.success(f"✅ Cleaned '{file_path.name}' → '{clean_path.name}'")

def download_dataset(urls, raw_data_path):
    raw_data_path = Path(raw_data_path)
    raw_data_path.mkdir(parents=True, exist_ok=True)
    
    for url in urls:
        logger.info(f"Downloading from {url}...")
        
        file_path = gdown.download(url, output=None, quiet=False, fuzzy=True, use_cookies=True)
        
        file_path = Path(file_path)
        target_path = raw_data_path / file_path.name
        file_path.rename(target_path)
        logger.success(f"✅ Downloaded to '{target_path}'")
        
        # clean up the review.json file
        clean_review_json(target_path)
        
        


    logger.info("Downloading dataset from Google Drive...")
    gdown.download(url, str(file_path), quiet=False, fuzzy=True, use_cookies=True)

    logger.success(f"✅ Downloaded JSON to '{file_path}'")

#if __name__ == "__main__":
#    # Use the direct download format — note the uc?id=... part
#    DATASET_URLS = ["https://drive.google.com/file/d/163KgvKNYPTV5_fArvlBiNkqb6rCDkk-Z/view?usp=drive_link",
#                    "https://drive.google.com/file/d/1O5kEcRxvnO1da74y9y5AE92h_QlOfjmL/view?usp=drive_link",
#                    "https://drive.google.com/file/d/1ox-qrD1sSKalbu25FIbu-kYGMvkn3Y3k/view?usp=drive_link", 
#                    "https://drive.google.com/file/d/1cJx0UUVxwsKL8DHpwrGoqBSlRA2v4I2f/view?usp=drive_link",
#                    "https://drive.google.com/file/d/10wpi7RzZTpt93_7uRZcshvA5X3ek3_Kc/view?usp=drive_links"]
#    
#    
#    RAW_DATA_PATH = Path("./data/raw")  # relative to where you run this script
#    
#download_dataset(DATASET_URLS, RAW_DATA_PATH)
