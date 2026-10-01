# 🚚 Logistics Strategic Planning & Data Analytics

### YuvaIntern Week 1 — Data-Driven Logistics Analysis

<p align="center">

<img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python">

<img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas">

<img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn">

<img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge">

<img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">

</p>

---

## 📌 Project Overview

This project was developed as part of the **YuvaIntern Week 1 internship task** to analyze logistics and supply-chain performance using a public logistics dataset.

The project follows an end-to-end data science workflow, starting from data cleaning and KPI development and progressing toward exploratory analysis, predictive modeling, clustering, and a strategic logistics optimization framework.

The main objective is to transform historical logistics data into meaningful insights that can support better operational planning and future decision-making.

### Core Areas

- Data Cleaning & Preprocessing
- KPI Development
- Exploratory Data Analysis
- Relationship Analysis
- Predictive Modeling
- Machine Learning
- Clustering
- Logistics Optimization
- Business Recommendations
- Strategic Planning

---

# 🎯 Business Problem

Modern logistics operations involve multiple factors that influence delivery performance, including:

- Shipping mode
- Scheduled shipping duration
- Actual shipping duration
- Order quantity
- Product value
- Discounts
- Customer segment
- Market
- Geographic region

The main business question addressed by this project is:

> **How can historical logistics data be analyzed to identify delivery-performance patterns, estimate shipping duration, segment logistics records, and support future route and resource optimization?**

---

# 🎯 Project Objectives

The project was developed with the following objectives:

1. Understand the structure and quality of the logistics dataset.
2. Clean and prepare the data for analysis.
3. Develop meaningful logistics KPIs.
4. Identify delivery-delay patterns.
5. Compare actual and scheduled shipping performance.
6. Analyze relationships between operational variables.
7. Build a baseline predictive model for shipping duration.
8. Segment logistics records using clustering.
9. Develop a conceptual logistics optimization framework.
10. Generate business-oriented recommendations.
11. Create a reproducible analytical workflow using Python.

---

# 📊 Dataset

The project uses a public **DataCo Supply Chain / Logistics dataset**.

The original dataset contains:

| Attribute | Value |
|---|---:|
| Original records | 180,519 |
| Variables | 53 |
| Records used for delivery analysis | 172,765 |
| Cancelled records excluded | 7,754 |

The dataset contains information related to:

| Category | Examples |
|---|---|
| Shipment | Actual shipping days, scheduled shipping days |
| Orders | Order ID, order date, order status |
| Products | Product category, product price |
| Customers | Customer segment |
| Geography | Region, market, country |
| Sales | Sales, discounts, order value |
| Delivery | Delivery status, late-delivery risk |
| Shipping | Shipping mode |

### Important Data Consideration

The analysis is performed at the **order-item record level**.

A single `Order ID` can contain multiple order-item records. Therefore, the number of records should not automatically be interpreted as the number of unique customer orders.

---

# 🧹 Data Preparation

The preprocessing stage included:

- Dataset inspection
- Data-type validation
- Missing-value analysis
- Duplicate checking
- Handling unusable fields
- Numeric variable preparation
- Categorical variable preparation
- Removal of cancelled records for delivery-performance analysis
- Preparation of datasets for machine-learning models

The cleaned dataset was then used consistently across the analytical workflow.

---

# 🔄 Project Workflow

```text
Raw Logistics Data
        │
        ▼
Data Inspection
        │
        ▼
Data Cleaning
        │
        ▼
KPI Development
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Relationship Analysis
        │
        ▼
Predictive Modeling
        │
        ▼
Clustering
        │
        ▼
Optimization Strategy
        │
        ▼
Business Recommendations