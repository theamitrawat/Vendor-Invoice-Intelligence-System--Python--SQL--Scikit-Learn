# 🧾 Vendor Invoice Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **ML-powered analytics platform for freight cost prediction, invoice risk detection, and vendor analytics**

---

## 🎯 **Project Overview**

An end-to-end machine learning system that automates vendor invoice processing workflows:

- **📦 Freight Cost Prediction** — Predict logistics costs from invoice features using regression models
- **🚨 Invoice Risk Detection** — Automatically flag suspicious invoices using classification models
- **📊 Interactive Dashboard** — Real-time analytics and predictions via Streamlit web app

---

## 🚀 **Live Demo**

🔗 **[View Live Application](https://your-app-url.streamlit.app)** *(Replace with your actual Streamlit Cloud URL)*

---

## 🛠️ **Tech Stack**

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **ML Framework** | Scikit-Learn |
| **Database** | SQLite |
| **Web Framework** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Deployment** | Streamlit Cloud |

---

## 📂 **Project Structure**

```
├── app.py                          # Main Streamlit dashboard
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
│
├── Data/
│   └── inventory.db                # SQLite database (404 MB, hosted on GitHub Releases)
│
├── models/
│   ├── predict_freight_model.pkl   # Trained regression model
│   ├── predict_flag_invoice.pkl    # Trained classification model
│   ├── scaler.pkl                  # Feature scaler
│   └── model_results.csv           # Model evaluation metrics
│
├── freight_cost_prediction/
│   ├── data_preprocessing.py       # Data loading & cleaning
│   ├── model_evaluation.py         # Model training & evaluation
│   └── train.py                    # Training pipeline
│
├── invoice_flagging/
│   ├── data_preprocessing.py       # Feature engineering for classification
│   ├── modeling_evaluation.py      # Classifier training
│   └── train.py                    # Training pipeline
│
├── inference/
│   ├── preddict_freight.py         # Freight prediction inference
│   └── predict_invoice_flag.py     # Risk detection inference
│
└── notebooks/
    ├── Predicting_Freight_Cost.ipynb
    └── Invoice_flagging.ipynb
```

---

## ⚙️ **Installation & Setup**

### **1. Clone Repository**

```bash
git clone https://github.com/theamitrawat/Vendor-Invoice-Intelligence-System--Python--SQL--Scikit-Learn.git
cd Vendor-Invoice-Intelligence-System--Python--SQL--Scikit-Learn
```

### **2. Create Virtual Environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Download Database**

The `inventory.db` file (404 MB) is hosted on GitHub Releases:

```bash
# Download manually from:
# https://github.com/theamitrawat/Vendor-Invoice-Intelligence-System--Python--SQL--Scikit-Learn/releases/download/v1.0/inventory.db

# Place it in: Data/inventory.db
```

*Note: The Streamlit app automatically downloads the database on first run when deployed to Streamlit Cloud.*

### **5. Run Application**

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🤖 **Machine Learning Pipeline**

### **Freight Cost Prediction (Regression)**

| Step | Description |
|------|-------------|
| **Data** | `vendor_invoice` table from SQLite |
| **Features** | `Quantity`, `Dollars` |
| **Target** | `Freight` |
| **Models** | Linear Regression, Decision Tree, Random Forest |
| **Evaluation** | RMSE, MAE, R² Score |
| **Best Model** | Linear Regression (RMSE: 124.66, R²: 0.963) |

### **Invoice Risk Detection (Classification)**

| Step | Description |
|------|-------------|
| **Data** | Join of `vendor_invoice` + `purchases` tables |
| **Features** | `invoice_quantity`, `invoice_dollars`, `Freight`, `total_item_quantity`, `total_item_dollars` |
| **Target** | `flag_invoice` (0 = Safe, 1 = Risky) |
| **Preprocessing** | StandardScaler normalization |
| **Model** | Random Forest Classifier (balanced class weights) |
| **Evaluation** | Accuracy, Precision, Recall, F1-Score |

---

## 📊 **Dashboard Features**

### **1. Dashboard** 📊
- KPI metrics (vendors, invoices, freight costs)
- Monthly trends & risk distribution
- Top vendors analysis
- Interactive scatter plots

### **2. Freight Prediction** 🚚
- Real-time cost prediction
- Sensitivity analysis charts
- Per-unit cost breakdown

### **3. Invoice Risk Detection** 🚨
- Risk probability scoring
- Discrepancy analysis
- Automated flagging

### **4. Data Explorer** 🔍
- Interactive table filtering
- Statistical summaries
- Correlation heatmaps
- Custom visualizations

### **5. Model Performance** 📈
- Model leaderboard
- Evaluation metrics comparison
- R² score gauge

### **6. About Project** ℹ️
- Technical documentation
- ML workflow timeline
- Tech stack details

---

## 🔧 **Training Models from Scratch**

### **Train Freight Prediction Model**

```bash
python -m freight_cost_prediction.train
```

### **Train Invoice Flagging Model**

```bash
python -m invoice_flagging.train
```

Models will be saved to `models/` directory.

---

## 🌐 **Deployment (Streamlit Cloud)**

1. Push code to GitHub
2. Upload `inventory.db` to GitHub Releases (tag: `v1.0`)
3. Deploy on [share.streamlit.io](https://share.streamlit.io)
4. App will auto-download database on first run

---

## 📈 **Model Performance**

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| **Linear Regression** | **124.66** | **23.76** | **0.963** |
| Random Forest | 130.92 | 27.54 | 0.959 |
| Decision Tree | 162.61 | 37.20 | 0.937 |

---

## 🤝 **Contributing**

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 **License**

This project is licensed under the MIT License.

---

## 👤 **Author**

**Amit Rawat**

- GitHub: [@theamitrawat](https://github.com/theamitrawat)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 🙏 **Acknowledgments**

- Dataset source: [Mention source if public]
- Inspired by real-world invoice processing challenges
- Built with ❤️ using Python & Streamlit

---

## 📧 **Contact**

For questions or feedback, reach out via [GitHub Issues](https://github.com/theamitrawat/Vendor-Invoice-Intelligence-System--Python--SQL--Scikit-Learn/issues).
