# Project name: Integrating Graph Clustering and Association Rule Mining on the Yelp Dataset for Detection of Global and Local Patterns

# TODO: 
- Finish the readme 
- Requiremnts.txt
- Add jupyter as pdf to overleaf 
- Go again through the report and re-read and re-read 
- Check the notebook and the .py files.
- Go through the captions in the figures and tables in the report and double check. 
- Check all the XX's.
## Setup :wrench:

**1. Clone the repository**

```bash
git clone https://github.com/Cerlog/02807-project
```

**2. Create conda environment** 
```bash
conda create -n 02807-project python=3.11 -y
```

**3. Activate the environment**
```bash 
conda activate 02807-project
```

**4. Change the directories** 
```bash
cd 02807-project
```
**5. Install the requirements**

```bash
pip install -r requirements.txt
```

## Project Structure :file_folder:



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
