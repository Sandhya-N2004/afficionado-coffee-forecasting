# ☕ Data-Driven Forecasting & Peak Demand Prediction
## Afficionado Coffee Roasters — New York City

## 📌 Project Overview
This project builds a multi-model demand forecasting system for Afficionado Coffee Roasters,
a multi-location coffee retail business operating 3 stores across New York City —
Lower Manhattan, Hell's Kitchen, and Astoria.

The system predicts daily and hourly sales per store, identifies peak demand periods,
and deploys all findings through an interactive Streamlit web dashboard.

## 🎯 Problem Statement
Afficionado Coffee Roasters lacked a demand forecasting system, causing overstocking,
understaffing, and lost revenue across its 3 store locations.

## ✅ Solution
Built a multi-model forecasting system using Python, XGBoost, Facebook Prophet,
ARIMA, and Exponential Smoothing to predict daily and hourly sales per store,
deployed as an interactive Streamlit dashboard.

## 📊 Dataset
- 149,116 transaction records
- 181 days (January to June 2025)
- 3 store locations
- 9 product categories
- Total Network Revenue: $698,812

## 🤖 Models Used
| Model | Accuracy |
|-------|----------|
| ARIMA | 92.2% ⭐ Best |
| Exponential Smoothing | 90.9% |
| Naive Forecast | 88.0% |
| XGBoost | 87.4% |
| Facebook Prophet | 86.5% |

## 🛠️ Technologies Used
- Python 3.12
- Pandas, NumPy, Scikit-learn
- Facebook Prophet, XGBoost, Statsmodels
- Plotly, Matplotlib, Seaborn
- Streamlit

## 📁 Project Files
- `app.py` — Streamlit dashboard application
- `Afficionado Coffee Roasters.csv` — Raw transaction dataset
- `Afficionado_Coffee_Complete_Code.ipynb` — Complete analysis notebook

## 👩‍💻 Author
Sandhya N (ENG22CS0439)
Department of Computer Science and Engineering
Dayananda Sagar University

Internship at Unified Mentor Pvt. Ltd., Haryana
Role: Data Analyst Intern
Duration: 01/02/2026 to 01/05/2026
