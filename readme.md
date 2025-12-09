# **EmbDI: Creating Embeddings of Heterogeneous Relational Datasets**

**SDSC5003 Project \- Research Paper Replication**

This repository contains the replication of the **EmbDI framework** (SIGMOD 2020), which performs Entity Resolution (ER) by transforming relational datasets into tripartite graphs and learning local embeddings.

## **👥 Group Information (Group ChatGPT6)**

| Student Name | Student ID |
| :---- | :---- |
| **Yu Le** | 59427131 |
| **Wang Qin** | 59741969 |
| **Lin Yumin** | 59847579 |
| **Li Chuang** | 59328000 |
| **Cai Yingyi** | 59736578 |

## **📂 Project Structure**

The project directory is organized as follows:
<pre>
5003_PROJECT/
├── EmbDI/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── edgelist.py
│   ├── entity_resolution.py
│   ├── main.py
│   └── utils.py
│
├── pipeline/
│   ├── config_files/
│   ├── datasets/
│   ├── dump/
│   ├── edgelists/
│   ├── embeddings/
│   ├── generated-matches/
│   ├── info/
│   ├── matches/
│   └── results/
│   └── walks/
</pre>


## **🛠️ Installation & Requirements**

We strongly recommend using **Anaconda** to manage the environment to avoid version conflicts (especially with gensim and scipy).

### **1\. Create a Virtual Environment**

conda create \-n embdi\_env python=3.9  
conda activate embdi\_env

### **2\. Install Dependencies**

Please install the specific versions listed below to ensure reproducibility:

pip install \-r requirements.txt

**Content of requirements.txt:**

gensim==4.3.2  
networkx==3.1  
tqdm==4.66.1  
pandas==2.0.3  
numpy==1.24.4  
scikit-learn  
scipy==1.10.1

*(Note: scipy==1.10.1 is crucial to prevent compatibility issues with Gensim)*

## **🚀 How to Run**

We provide a helper script run.py to automatically handle path configurations.

### **Method 1: One-Click Run (Recommended)**

You can run this command from the root directory (5003\_project/). By default, it runs the fodors\_zagats dataset.

python run.py

### **Method 2: Standard Command Line**

If you prefer running the module directly, ensure you are in the root directory:

\# Run Fodors-Zagats  
python EmbDI/main.py \-f pipeline/config\_files/config-fodors.ini

\# Run Beer Dataset  
python EmbDI/main.py \-f pipeline/config\_files/config-beer.ini

\# Run DBLP-ACM  
python EmbDI/main.py \-f pipeline/config\_files/config-dblp.ini

## **📊 Experimental Results**

We successfully replicated the results on three benchmark datasets. The results (F1 Score) align with the original paper.

| Dataset | Precision | Recall | F1 Score | Status |
| :---- | :---- | :---- | :---- | :---- |
| **Fodors-Zagats** | \~99.1% | \~99.1% | **99.1%** | ✅ Exact Match |
| **DBLP-ACM** | \~99.5% | \~98.2% | **98.8%** | ✅ Exact Match |
| **BeerAdvo** | \~93.6% | \~86.8% | **90.1%** | ⚠️ Expected Drop\* |

*\*Note: As analyzed in our report, the Beer dataset shows a performance drop due to high sparsity and missing nodes, which affects the random walk quality.*

## **⚙️ Configuration Details**

Configuration files are located in pipeline/config\_files/. Key parameters used for replication:

* embedding\_dim: **300**  
* walk\_length: **60**  
* num\_walks: **30**  
* window\_size: **5**  
* learning\_method: **skipgram**

## **📄 Reference**

* **Original Paper:** *Creating Embeddings of Heterogeneous Relational Datasets for Data Integration Tasks* (SIGMOD 2020).  
* **Report:** See Group-ChatGPT6-Report.pdf for detailed analysis and methodology.
