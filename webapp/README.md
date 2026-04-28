# Vehicle Density Detection and Route Optimisation

> A machine learning-based framework for vehicle density detection and route optimisation.
> B.Sc. Computer Science Final Year Project — Federal University Oye-Ekiti (FUOYE), 2026.

---

## 👤 Author

**Sanusi Moshood Olanrewaju**  
Matriculation Number: FTP/CSC/25/01116478  
Supervisor: Mr. Onadokun Isaac Olawale  
Department of Computer Science, FUOYE

---

## 📌 Project Overview

This project proposes and implements a machine learning-based framework that:

1. **Detects vehicle density** on road segments (Low / Medium / High) using supervised ML
2. **Optimises routes** using Dijkstra's algorithm with density-weighted edge penalties
3. **Provides a web interface** where users enter a source, destination, and time — and receive the best route, congestion level, and estimated travel time

### System Architecture

```
INPUT → PREPROCESSING → ML PREDICTION → ROUTE OPTIMISATION → OUTPUT
```

- **ML Models:** Decision Tree (97.75%) and Random Forest (97.50% accuracy)
- **Dataset:** Metro Interstate Traffic Volume — UCI ML Repository (48,187 records)
- **Routing:** Dijkstra's Algorithm with density-penalised edge weights
- **Interface:** Flask web application

---

## 🗂 Project Structure

```
VehicleDensityProject/
│
├── webapp/                         ← Flask web application
│   ├── app.py                      ← Main Flask app
│   ├── templates/index.html        ← Web interface
│   ├── requirements.txt            ← Python dependencies
│   ├── Procfile                    ← Render deployment config
│   └── render.yaml                 ← Render service config
│
├── data/
│   ├── Metro_Interstate_Traffic_Volume.csv   ← Raw dataset (UCI)
│   └── clean_traffic_data.csv               ← Preprocessed dataset
│
├── models/
│   ├── decision_tree_model.pkl     ← Trained Decision Tree
│   ├── random_forest_model.pkl     ← Trained Random Forest
│   └── feature_cols.pkl            ← Feature column names
│
├── outputs/                        ← Generated charts and results
│   ├── Step1_Exploratory_Analysis.png
│   ├── Step2_Confusion_Matrices.png
│   ├── Step2_Performance_Comparison.png
│   ├── Step2_Feature_Importance.png
│   ├── Step2_CrossValidation.png
│   ├── Step3_Road_Network_Density.png
│   ├── Step3_Route_Comparison.png
│   ├── Step3_Scenarios_Summary.png
│   ├── Step4_System_Pipeline.png
│   ├── Step4_Query_Performance.png
│   ├── Step4_System_Dashboard.png
│   ├── model_results.json
│   ├── routing_results.json
│   └── integration_summary.json
│
├── Step1_Data_Preparation.py       ← Data preprocessing pipeline
├── Step2_Model_Training.py         ← ML model training & evaluation
├── Step3_Route_Optimisation.py     ← Graph routing module
└── Step4_System_Integration.py     ← Full system integration
```

---

## ⚙️ Installation and Setup

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/VehicleDensityProject.git
cd VehicleDensityProject
```

### 2. Install dependencies
```bash
pip install -r webapp/requirements.txt
```

### 3. Run the data pipeline (if running from scratch)
```bash
python3 Step1_Data_Preparation.py
python3 Step2_Model_Training.py
python3 Step3_Route_Optimisation.py
python3 Step4_System_Integration.py
```

### 4. Launch the web application
```bash
cd webapp
python3 app.py
```

Then open your browser and go to:
```
http://localhost:5000
```

---

## 🌐 Live Demo

The web application is deployed on Render:  
🔗 **[https://vehicle-density-optimizer.onrender.com](https://vehicle-density-optimizer.onrender.com)**

> Note: The free tier may take 30–60 seconds to wake up on first visit.

---

## 📊 Results Summary

| Metric | Decision Tree | Random Forest |
|--------|--------------|---------------|
| Test Accuracy | 97.75% | 97.50% |
| Macro Precision | 97.74% | 97.50% |
| Macro Recall | 97.75% | 97.50% |
| Macro F1-Score | 97.74% | 97.50% |
| CV Accuracy (Mean) | 96.14% | 96.92% |
| CV Std Dev | ±0.52% | ±0.38% |

**Route Optimisation:**
- Routes improved: 3 out of 5 (60%)
- High-density segments avoided: 4
- Average distance overhead: 7.3%

---

## 🛠 Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Primary programming language |
| Scikit-learn | Decision Tree & Random Forest models |
| NetworkX | Road graph & Dijkstra's algorithm |
| Pandas / NumPy | Data processing |
| Matplotlib / Seaborn | Visualisations |
| Flask | Web application framework |
| Gunicorn | WSGI server for deployment |

---

## 📄 Dataset

**Metro Interstate Traffic Volume**  
Source: UCI Machine Learning Repository  
URL: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume  
Records: 48,204 (48,187 after cleaning)  
Features: 8 engineered input features  
Target: Vehicle Density Level (Low / Medium / High)

---

## 📝 Licence

This project is submitted in partial fulfilment of the requirements for the award of the Bachelor of Science (B.Sc.) degree in Computer Science at the Federal University Oye-Ekiti (FUOYE). All rights reserved © 2026.
