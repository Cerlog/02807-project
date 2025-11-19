from pathlib import Path
from loguru import logger
import subprocess
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

reviews = "data/raw/review.json"

clean_review_json(reviews)
