# Integrating Graph Clustering and Association Rule Mining on the Yelp Dataset for Detection of Global and Local Patterns

## Project Overview 
This project explores the integration of graph clustering techniques with association rule mining to uncover both global and local patterns in the Yelp user dataset. We leverage the Louvain algorithm for community detection and the Apriori algorithm for mining association rules.  

## Project Structure :file_folder:
```
.
├── data/
│   ├── processed/          # Processed data files
└──── raw/                  # Raw dataset files
├── figures/                # Generated figures and plots
├── src/                    # Source code
│   ├── apriori.py          # Association rule mining implementation
│   ├── data.py             # Data downloading and cleaning utilities
│   ├── functions.py        # Helper functions
│   └── preprocessing.py    # Data preprocessing scripts
├── FINAL_NOTEBOOK.ipynb    # Analysis notebook
├── readme.md               # Project documentation
└── requirements.txt        # Python dependencies
```

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

## Usage :computer:
### 1. Data Downloading and Preprocessing
The dataset is too loarge, to be stored on GitHub. It is therefore hosted on Google Drive and can be downloaded by running the main notebook. The dataset we use is the Yelp Open Dataset, which can be found here: https://www.yelp.com/dataset.

### 2. Running the Analysis
The entire analysis can be run in the `FINAL_NOTEBOOK.ipynb` notebook. This notebook includes data preprocessing, associaton rules, graph clustering and community detection, apriori on communities and sentiment analysis. 

