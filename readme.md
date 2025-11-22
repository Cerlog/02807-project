# Project name: Pending

!TODO: ADD PROJECT DESCRIPTION 

The overall description of the project so far: 

- Run the apriori algorithm on the Yelp dataset to generate: 
    - positive 
    - negative rules 
- Use the graph networks to locate communities within the dataset 
  - on the following communities apply the Appriori algortihm to discover the rules to local communities
  - analyze these with sentiment analysis to further understand the communities

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

The project structure:
!TODO:


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
