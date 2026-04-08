# Customer Segmentation using K-Means

## Overview

This project implements a customer segmentation pipeline using **K-Means clustering** to identify distinct groups of customers based on their attributes. The objective is to enable **data-driven marketing strategies and customer targeting**.

The workflow includes data preprocessing, clustering, and model persistence for reuse.

---

## Key Features

* Unsupervised learning using K-Means clustering
* Feature scaling for improved clustering performance
* Modular project structure with separation of data, models, and code
* Model persistence for reproducible results

---

## Project Structure

```
.
├── data/
│   └── customer_segmentation.csv     # Dataset
│
├── models/
│   ├── kmeans.pkl                   # Trained clustering model
│   └── scaler.pkl                   # Preprocessing scaler
│
├── customer_segmentation.ipynb      # EDA and experimentation
├── segmentation.py                  # Main pipeline script
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset contains customer-level features used for segmentation. These may include:

* Demographics (e.g., age)
* Financial attributes (e.g., income)
* Behavioral metrics (e.g., spending patterns)

---

## Methodology

### 1. Data Preprocessing

* Handling missing values (if applicable)
* Feature scaling using standardization (`scaler.pkl`)

### 2. Clustering

* Applied **K-Means algorithm**
* Optimal number of clusters determined using:

  * Elbow Method
  * (Optional) Silhouette Score

### 3. Model Persistence

* Trained model saved as `models/kmeans.pkl`
* Scaler saved as `models/scaler.pkl`

---

## Installation

Clone the repository:

```
git clone https://github.com/MainakJ99/customer-segmentation.git
cd customer-segmentation
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

Run the segmentation pipeline:

```
python segmentation.py
```

---

## Results

* Customers are grouped into distinct clusters
* Each cluster represents a segment with similar characteristics

(*Add cluster visualizations or metrics if available*)

---

## Author

Mainak Jana
