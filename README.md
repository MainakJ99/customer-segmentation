Customer Personality Analysis — Segmentation

Customer segmentation project using K-Means, Hierarchical Clustering, and DBSCAN on the Customer Personality Analysis dataset.

Results
Model	Clusters	Silhouette Score	Davies-Bouldin Index	Dunn Index
K-Means	2	0.222	1.736	0.065
Hierarchical	2	0.198	1.838	0.076
DBSCAN	3	0.232	1.396	0.350

Final model used: K-Means (k=2)

Customer Segments
High-Engagement Premium Customers
Higher income
Higher spending
More campaign acceptance
More store and catalog purchases
Budget-Conscious Family Customers
Lower income
Lower spending
More children at home
Higher web visits but lower conversion
Folder Structure
customer-segmentation/
│
├── app.py
├── customer_segmentation.ipynb
├── requirements.txt
├── README.md
│
├── data/
│   └── customer_segmentation.csv
│
├── models/
│   ├── scaler.pkl
│   └── model_kmeans.pkl
│
└── plots/
Setup

Clone the repository:

git clone https://github.com/MainakJ99/customer-segmentation.git
cd customer-segmentation

Create virtual environment:

python -m venv env

Activate environment:

Windows
env\Scripts\activate
Mac/Linux
source env/bin/activate

Install dependencies:

pip install -r requirements.txt
Run Notebook
jupyter notebook customer_segmentation.ipynb
Run Streamlit App
streamlit run app.py

App runs on:

http://localhost:8501
Tech Stack
Python
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
streamlit
joblib
Dataset

Customer Personality Analysis Dataset from Kaggle.

Author

Mainak Jana
