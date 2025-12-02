import gdown
import subprocess
from pathlib import Path
from loguru import logger
from typing import List, Union

def clean_review_json(file_path: Union[str, Path]) -> None:
    """
    Clean the 'yelp_academic_dataset_review.json' file.
    
    This function fixes formatting issues, such as unclosed brackets, by converting
    it to a clean ndjson format using jq.

    Args:
        file_path: Path to the raw review JSON file.
    """
    file_path = Path(file_path)
    if file_path.name != "yelp_academic_dataset_review.json":
        logger.warning("Can't find 'yelp_academic_dataset_review.json' to clean.")
        return 
    
    clean_path = file_path.with_name("review_clean.ndjson")
    
    logger.info("Cleaning 'yelp_academic_dataset_review.json'...")
    
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

def download_dataset(urls: List[str], raw_data_path: Union[str, Path]) -> None:
    """
    Download dataset files from Google Drive URLs.

    Args:
        urls: List of Google Drive URLs.
        raw_data_path: Directory to save the downloaded files.
    """
    raw_data_path = Path(raw_data_path)
    raw_data_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading dataset from Google Drive...")

    for url in urls:
        logger.info(f"Downloading from {url}...")
        
        # Download the file
        file_path = gdown.download(url, output=None, quiet=False, fuzzy=True, use_cookies=True)
        
        if file_path:
            file_path = Path(file_path)
            target_path = raw_data_path / file_path.name
            file_path.rename(target_path)
            logger.success(f"✅ Downloaded to '{target_path}'")
            
            # clean up the review.json file if it's the one we just downloaded
            if target_path.name == "yelp_academic_dataset_review.json":
                clean_review_json(target_path)
        else:
            logger.error(f"Failed to download from {url}")

if __name__ == "__main__":
    # Use the direct download format — note the uc?id=... part
    DATASET_URLS = [
        "https://drive.google.com/file/d/163KgvKNYPTV5_fArvlBiNkqb6rCDkk-Z/view?usp=drive_link",
        "https://drive.google.com/file/d/1O5kEcRxvnO1da74y9y5AE92h_QlOfjmL/view?usp=drive_link",
        "https://drive.google.com/file/d/1ox-qrD1sSKalbu25FIbu-kYGMvkn3Y3k/view?usp=drive_link", 
        "https://drive.google.com/file/d/1cJx0UUVxwsKL8DHpwrGoqBSlRA2v4I2f/view?usp=drive_link",
        "https://drive.google.com/file/d/10wpi7RzZTpt93_7uRZcshvA5X3ek3_Kc/view?usp=drive_links"
    ]
    
    RAW_DATA_PATH = Path("./data/raw")  # relative to where you run this script
    download_dataset(DATASET_URLS, RAW_DATA_PATH)
