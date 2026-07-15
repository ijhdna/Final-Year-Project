# Student Mental Health Detection Dashboard using Hybrid Sentiment Approach

An automated screening and monitoring solution developed for Universiti Teknologi PETRONAS (UTP) to address social desirability bias in student mental health reporting. This system integrates quantitative clinical metrics (DASS-21) with qualitative natural language processing (NLP) to flag students who may be masking psychological distress.

---

## 📌 Project Overview
Traditional university mental health screenings rely on standard clinical surveys (such as the DASS-21), which are highly prone to "masking" or social desirability bias. Students often under-report their distress on multiple-choice scales to avoid stigma. 

This project implements a **Hybrid Sentiment Analysis Engine** that pairs quantitative survey data with unstructured, qualitative student confessions. The system processes both inputs to predict a more accurate depression severity level and exports the results to an interactive **Power BI Dashboard** designed for UTP's Psychological and Counselling Service (PCS).

---

## ⚙️ Architecture & Methodology
The system follows a 6-stage pipeline:
1. **Data Collection**: Dual-input gathering of quantitative DASS-21 scores and qualitative text confessions.
2. **Data Preprocessing**: Text standardization via Tokenization, Lemmatization, and Stopword Removal.
3. **Feature Engineering**: Hybrid feature matrix combination using **VADER** (Valence Aware Dictionary and Sentiment Reasoner) sentiment scores (-1 to +1) and **TF-IDF Vectorization** for machine learning features.
4. **Modelling & Training**: Training and evaluation of supervised classifiers (Random Forest, Support Vector Machines, and Naïve Bayes) using **12-Fold Cross-Validation**.
5. **Evaluation & Validation**: Testing the models against unseen, real-world validation data. **Random Forest** is selected as the primary architecture.
6. **Reporting & Dashboarding**: Exporting predictions to a centralized Power BI dashboard for counselors.

---

## 📊 Key Results & Performance
* **Training Phase (Synthetic Dataset)**: 
  * **Random Forest**: **99.5% Accuracy** | 99.7% Precision | 99.4% Recall | 99.5% F1-Score.
  * **SVM**: **97.1% Accuracy**.
  * **Naïve Bayes**: **84.3% Accuracy**.
* **Validation Phase (Unseen, Real UTP Student Responses)**:
  * Retained a reliable **73.08% Accuracy** on blind test data.
  * Successfully flagged **7 key mismatches** where students masked their distress (e.g., scoring "Severe" on the quantitative scale but writing a highly controlled, mild confession narrative).

---

## 🛠️ Tech Stack & Tools
* **Language**: Python 3.x
* **IDE**: Visual Studio Code
* **Core Libraries**: `pandas`, `scikit-learn`, `nltk` (VADER/Lemmatizer), `numpy`
* **Automation**: Microsoft Power Automate (syncing surveys from Microsoft Forms to Excel)
* **Visualization**: Power BI (Centralized Mental Health Dashboard)

---

## 📂 Repository Structure
```text
├── data/
│   ├── DASS_Dummy.csv                      # Dummy dataset that follows PCS UTP Database structure
│   ├── Synthetic_Dataset_From_Dummy.csv    # Dataset used for model training & testing
│   ├── data benchmark.xlsx                 # Unseen validation dataset from UTP students
│   └── Validation_Predictions_PowerBI.csv  # Unseen validation dataset from UTP students
├── models/
│   └── hybrid_engine.pkl                   # Trained Best model export
├── notebooks_or_scripts/
│   ├── synthetic_data_creation.py          # Developing the synthetic data
│   ├── preprocess_feat-eng_senti.py        # Pre-processing & Feature Engineering using SentiWordNet
│   ├── preprocess_FE_vader                 # Pre-processing & Feature Engineering using VADER
│   ├── modelling                           # Training and Testing code
│   └── validation.py                       # Validation by using the hybrid engine with the validation dataset
├── dashboard/
│   └── UTP_Mental_Health_Dashboard.pbix    # UTP Student Mental Health Dashboard Power BI  
├── README.md
└── requirements.txt                        # Python dependencies
