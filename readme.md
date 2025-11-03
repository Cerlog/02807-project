# Project name: Pending


# Todo
- join the 'user.json' together with the 'business.json' and 'review.json'





## Dataset :package:

The project uses the **Yelp user dataset** (!TODO add link to the dataset), stored under the `data/` directory with the following structure:

```
data/
├── raw/
│   └── dataset.json
└── processed/
```

The script `data.py` automatically downloads the dataset from Google Drive and places it in `data/raw`, as it is too large to be on GitHub.


Before running the script, install the required dependencies:

```bash
pip install -r requirements.txt
```

Then download the dataset:

```bash
python data.py
```

After downloading, the raw JSON file can be preprocessed and saved in data/processed for further analysis.
