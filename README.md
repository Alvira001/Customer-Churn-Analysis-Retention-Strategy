# 📊 Customer Churn Analysis & Retention Strategy

This end-to-end data analytics project explores customer churn behavior in a telecom company using Python, SQL, Excel, Power BI, and machine learning. It provides insights into churn drivers and proposes data-backed strategies for improving retention.

---

## 🧠 Project Goals

- Identify key drivers of customer churn
- Build a machine learning model to predict churn
- Create dashboards to support business decision-making
- Demonstrate real-world data analysis skills across multiple tools

---

## 🛠️ Tools & Technologies

| Tool              | Purpose                                           |
|-------------------|---------------------------------------------------|
| **Excel**         | Initial exploration and manual cleaning           |
| **SQL (SQLite)**  | Data querying and schema simulation               |
| **Python**        | Data wrangling, visualization, modeling           |
| **pandas**        | Data transformation and manipulation              |
| **scikit-learn**  | Predictive modeling and evaluation                |
| **matplotlib/seaborn** | Exploratory data visualization             |
| **Power BI**      | Dashboard development for stakeholder insights    |
| **Git/GitHub**    | Version control and portfolio showcase            |

---

## 📁 Project Structure
customer_churn_analysis/
│
├── data/
│ └── telco_churn.csv ← [Download instructions below]
├── churn_analysis.py ← Main Python script
├── churn_predictions.csv ← Output for BI tools
├── feature_importance.png ← Top 10 churn drivers chart
├── requirements.txt ← Python dependencies
└── README.md ← This file

---

## 📦 Dataset (Required)

This project uses the **Telco Customer Churn** dataset from Kaggle:

🔗 [Download the Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

After downloading:
- Rename the file to: `telco_churn.csv`
- Move it to the `data/` folder in this project

---

## 🚀 How to Run

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/customer_churn_analysis.git
   cd customer_churn_analysis

Install the required Python packages
```bash
pip install -r requirements.txt

Download the dataset and place it in
data/telco_churn.csv

Run the analysis
```bash
python churn_analysis.py

Explore churn_predictions.csv in Power BI to build dashboards

