# Credit Risk Analysis - LendingClub Project

An end-to-end Machine Learning project designed to assess financial risk. This project utilizes the **LendingClub** dataset to build a predictive model that identifies potential loan defaults, providing actionable insights through an interactive dashboard.

## Author Information
* **Name:** Nguyen Hoang Hong Ngoc
* **Major:** Information Technology (IT)
* **Institution:** Ho Chi Minh City University of Education (HCMUE)

---

## Project Structure
The project follows a modular structure to separate data, logic, and experimentation:
Credit-risk-Analysis/
├── app/                     # Multi-page Dashboard application
│   ├── pages/
│   │   ├── 1_Overview.py    # General statistics and metrics
│   │   └── 2_Analysis.py    # Detailed risk analysis charts
│   └── app.py               # Main entry point for the Streamlit app
├── data/                    # Data management (Ignored by Git)
│   ├── processed/           # Cleaned and engineered data
│   │   └── clean_loan.csv
│   └── raw/                 # Original datasets from Kaggle
│       ├── accepted_2007_to_2018Q4.csv
│       ├── rejected_2007_to_2018Q4.csv
│       └── lending-club.zip
├── models/                  # Saved ML models and preprocessors
│   ├── background.pkl       # Visual assets/background data
│   ├── feature_names.pkl    # List of features used in training
│   └── logistic_model.pkl   # Trained Logistic Regression model
├── notebooks/               # Research & Development
│   └── eda.ipynb            # Data cleaning and exploratory analysis
├── src/                     # Core logic scripts
│   ├── credict.py           # Legacy dashboard script/utility functions
│   └── train_model.py       # Script for model training and evaluation
├── venv/                    # Virtual Environment
├── .gitignore               # Configuration to exclude large/private files
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```


## Installation & Setup

### 1. Environment Setup
Clone the repository and set up a clean Python environment:
```powershell
git clone https://github.com/k-hgnoc/Credit-risk-Analysis
cd Credit-risk-Analysis
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Preparation
1.  Download the dataset from [Kaggle - Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
2.  Extract the files into `data/raw/`.
3.  Ensure the file names match: `accepted_2007_to_2018Q4.csv`.

---

## 🚀 Execution Flow

### Step 1: Exploratory Data Analysis
Review `notebooks/eda.ipynb` to understand the data distribution, correlation, and the preprocessing steps taken to handle missing values and outliers.

### Step 2: Model Training
Execute the training script to regenerate models if needed:
```powershell
python src/train_model.py
```
*The trained model will be saved to the `models/` directory.*

### Step 3: Run the Dashboard
Launch the multi-page Streamlit application:
```powershell
streamlit run app/app.py
```

---

## 🧪 Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (Logistic Regression, Feature Scaling)
* **Visualization:** Matplotlib, Seaborn, Streamlit
* **Deployment:** Streamlit (Multi-page App support)

---

## 📝 Maintenance
To update the dependencies after installing new libraries:
```powershell
pip freeze > requirements.txt
```
