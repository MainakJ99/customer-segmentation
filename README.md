# Customer Segmentation using K-Means

## Overview

This project performs customer segmentation using **K-Means clustering** to group customers based on their attributes. The workflow includes exploratory analysis, model training, and clustering-based insights.

---

## Key Features

* K-Means clustering for unsupervised segmentation
* Feature scaling using standardization
* Model persistence using saved `.pkl` files
* Separation of experimentation (notebook) and execution (script)

---

## Project Structure

```id="k2m9zt"
.
├── data/
│   └── customer_segmentation.csv
│
├── models/                        # Generated locally (not tracked)
│   ├── kmeans.pkl
│   └── scaler.pkl
│
├── customer_segmentation.ipynb   # Model training and analysis
├── segmentation.py               # Uses saved models for segmentation
├── requirements.txt
└── README.md
```

---

## Important Note

The trained model (`kmeans.pkl`) and scaler (`scaler.pkl`) are **not included in the repository**.

They must be generated before running the script.

---

## Setup & Usage

### Step 1: Generate Models (Required)

Run the Jupyter Notebook:

```bash
jupyter notebook customer_segmentation.ipynb
```

Execute all cells to generate:

* `models/kmeans.pkl`
* `models/scaler.pkl`

---

### Step 2: Run Segmentation Script

```bash
python segmentation.py
```

---

## Methodology

### Data Preprocessing

* Feature scaling using standardization

### Clustering

* K-Means algorithm applied
* Number of clusters chosen using:

  * Elbow Method
  

---

## Results

* Customers are grouped into distinct clusters
* Each cluster represents a segment with similar characteristics

---


## Author

Mainak Jana
