
# 🚀 Automated Trading System: Real-Time Stock Analysis Dashboard

This project is a real-time stock analysis and trading strategy dashboard built using **Streamlit**, designed to explore U.S. equities data, visualize price trends, and simulate trading strategies with machine learning models.

> 📈 Powered by SimFin’s live API and custom ML models for predictive analytics.

---

## 📂 Project Structure

```
automated-trading-system/
├── streamlit_app.py              # Main Streamlit dashboard
├── src/
│   ├── etl.py                    # Extract, Transform, Load logic
│   ├── model.py                  # ML model training & loading
│   ├── backtesting.py            # Backtest strategy logic
│   ├── optimized_model.joblib    # Trained ML model
│   └── pysimfin.py               # SimFin API wrapper
├── data/
│   ├── raw/                      # Input data (e.g., CSVs, ZIPs)
│   └── processed/                # Cleaned datasets
├── .streamlit/
│   └── secrets.toml              # API key for SimFin (Streamlit Cloud)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🌟 Key Features

- 📊 **Real-Time Price Analysis** with live SimFin API data  
- 🧠 **ML-Powered Predictions** using XGBoost and Scikit-learn  
- 🧪 **Backtesting Framework** to simulate strategy performance  
- 🖼️ **Streamlit Dashboard** for interactive exploration  
- 🔐 Secure API key handling via `secrets.toml`

---

## ⚙️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/eminabrahamian/Automated-Trading-System.git
cd Automated-Trading-System
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Dashboard

### Locally
```bash
streamlit run streamlit_app.py
```

> The app will launch in your browser at `http://localhost:8501`

---

## 🔐 API Key Setup

To use SimFin’s API, you’ll need an API key.

### For local use:
Create a `.env` file and add:
```
API_KEY=your_simfin_key_here
```

### For Streamlit Cloud:
Create `.streamlit/secrets.toml` and add:
```toml
API_KEY = "your_simfin_key_here"
```

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python (Pandas, NumPy, Requests)  
- **ML Models**: XGBoost, Scikit-learn, Joblib  
- **Data Source**: SimFin API  
- **Visualization**: Matplotlib, Seaborn  

---

## 📌 Future Improvements

- Add user authentication for private portfolios  
- Save trade logs and results to cloud storage  
- Implement real-time alerts or email notifications  
- Dockerize the app for deployment anywhere

---

## 🤝 Credits
IE MBD Term 2 – Group 7
Built as part of IE MBD coursework (Python for Data Analysis II)
