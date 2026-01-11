# Customer Genome Intelligence  
A Full-Stack Customer Behavior Intelligence & Decision System

---

## Overview

Customer Genome Intelligence is an end-to-end data science system designed to transform raw or semi-structured customer data into **actionable business intelligence**.
VIDEO LINK = https://www.linkedin.com/posts/anurag-kumar-singh4440_customer-genome-intelligence-built-a-customer-activity-7415797799054778368-P0ZX?utm_source=share&utm_medium=member_desktop&rcm=ACoAAD2NwdkBVynk1_PhoRha6EEqB01AQC-0U50
Instead of stopping at clustering or prediction, this system focuses on:
- Behavioral understanding
- Risk-aware decision making
- Confidence-driven recommendations
- Practical actions that can be executed in real business environments

The application is implemented as an interactive Streamlit dashboard backed by machine learning models trained on customer behavioral features.

---

## Core Problem This System Solves

Most customer analytics projects fail because they:
- Stop at clustering without interpretation
- Predict churn without confidence estimation
- Lack business-aligned actions
- Break when real-world data schema changes

This system is built to address those exact gaps.

<img width="339" height="802" alt="image" src="https://github.com/user-attachments/assets/33074f61-6d9d-40a0-839a-60ebcc0da511" />
<img width="1919" height="1079" alt="Screenshot 2026-01-10 220100" src="https://github.com/user-attachments/assets/5be1cc12-cc43-4bbd-8d0c-047b7768d59f" />
<img width="1919" height="1012" alt="Screenshot 2026-01-10 215404" src="https://github.com/user-attachments/assets/8692202d-922d-457d-a0c2-9786d629cd7f" />





---

## High-Level System Flow

| Stage | Description |
|-----|------------|
| Data Ingestion | Accepts customer-level aggregated CSV |
| Feature Standardization | Ensures correct feature ordering and scaling |
| Behavioral Clustering | Groups customers based on behavioral similarity |
| Value Prediction | Estimates customer future value |
| Risk Assessment | Computes churn or instability probability |
| Confidence Scoring | Measures reliability of predictions |
| Similarity Analysis | Finds behaviorally similar customers |
| Strategy Engine | Generates cluster-aware business actions |
| Visualization | Interactive dashboard with charts and tables |
| Deployment | Streamlit-based web application |


---

## 🔄 Analytical Workflow

---

## Accepted Input Data Format

The system expects **customer-level aggregated data**, not raw transactions.

### Required Columns

| Column Name | Description |
|-----------|------------|
| CustomerID | Unique customer identifier |
| total_orders | Total number of orders |
| total_quantity | Total items purchased |
| total_spend | Total monetary spend |
| avg_order_value | Average spend per order |
| recency_days | Days since last purchase |
| unique_products | Number of distinct products purchased |
| cluster | Optional (auto-generated if missing) |

Column names are normalized internally, so casing and spacing differences are handled automatically.

---

## Feature Schema Used by Models

All models are trained and inferenced using the same ordered feature set.

| Feature | Purpose |
|------|--------|
| total_orders | Purchase frequency |
| total_quantity | Volume indicator |
| total_spend | Customer value |
| avg_order_value | Spending behavior |
| recency_days | Engagement freshness |
| unique_products | Diversity of interest |

Maintaining this exact order is critical for model consistency.

---

## Machine Learning Components

### 1. Behavioral Clustering
- Algorithm: KMeans
- Input: Scaled behavioral features
- Purpose: Discover natural customer segments

### 2. Value Prediction Model
- Type: Regression model
- Output: Estimated customer value
- Used for prioritization and ROI planning

### 3. Stability / Risk Model
- Type: Probabilistic classifier
- Output: Risk probability (0–1)
- Converted into confidence score

### 4. Confidence Score
#### **confidence = 1-risk_probability**


This allows decisions to be made based on reliability, not just predictions.

---

## Customer Segment Interpretation

Each cluster is mapped to a business-meaningful label:

| Cluster | Segment Name | Description |
|-------|-------------|------------|
| 0 | Loyal & High Value | Stable, high-revenue customers |
| 1 | Growing Customers | Increasing engagement and spend |
| 2 | Price Sensitive | Responsive to discounts |
| 3 | At Risk | Declining engagement |
| 4 | High Churn Risk | Immediate intervention required |

These labels are **presentation-level only** and do not affect model logic.

---

## Decision & Action Strategy Engine

The system does not just analyze; it recommends actions.

| Segment | Recommended Strategy |
|------|----------------------|
| Loyal & High Value | Loyalty programs, premium services |
| Growing Customers | Cross-sell, upsell, personalization |
| Price Sensitive | Discounts, bundles |
| At Risk | Re-engagement campaigns |
| High Churn Risk | Direct outreach, win-back offers |

Actions are shown prominently and written in business language.

---

## Dashboard Capabilities

### Executive Overview
- Total customers
- Average spend
- Average orders
- Number of segments

### Visual Analytics
- Pie chart showing cluster distribution with count and percentage
- Bar charts showing spend per cluster

### Individual Customer Intelligence
- Cluster assignment
- Predicted value
- Risk percentage
- Confidence score
- Similar customer identification
- Action recommendation

### Cluster-Level Analysis
- Per-cluster statistics
- Sample customers per cluster
- Strategic insights

---

## Similarity Analysis

The system computes cosine similarity between customers in feature space.

This allows:
- Peer comparison
- Look-alike modeling
- Pattern-based recommendations

---

## Export & Reporting

The final intelligence report can be downloaded as CSV and includes:
- Customer features
- Cluster label
- Predicted value
- Risk probability
- Confidence score
- Strategy context

This enables offline analysis and stakeholder sharing.

---

## Technology Stack

| Layer | Tools |
|-----|------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| ML | scikit-learn |
| Visualization | Matplotlib, Streamlit |
| Model Persistence | joblib |
| Deployment | Streamlit |

---

## Deployment Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

streamlit run app.py
