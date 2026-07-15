import pandas as pd
import re
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score

# NLP assets 
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

validation_excel_path = r"C:\Users\Izzah\Documents\3. FYP\src\data benchmark.xlsx"

try:
    raw_df = pd.read_excel(validation_excel_path)
    print(f"[LOAD] Unseen validation dataset loaded successfully from Excel. Total rows: {len(raw_df)}")
except FileNotFoundError:
    print(f"Error: Target validation file not found at {validation_excel_path}")
    exit()

# =====================================================================
# 1. FUZZY COLUMN DETECTION ENGINE
# =====================================================================
try:
    raw_df = pd.read_excel(validation_excel_path)
    print(f"[LOAD] Unseen validation dataset loaded successfully from Excel. Total rows: {len(raw_df)}")
except FileNotFoundError:
    print(f"Error: Target validation file not found at {validation_excel_path}")
    exit()

# Lowercase column mapping dictionary to resolve dynamic variations
column_pool = {col.lower(): col for col in raw_df.columns}

def find_dynamic_column(keywords, default_name):
    """Scans the loaded dataframe headers for partial keyword matches."""
    for kw in keywords:
        for low_col in column_pool:
            if kw in low_col:
                return column_pool[low_col]
    return None

# Perform fuzzy mapping matching based on current parameters
mapped_id = find_dynamic_column(['studentid', 'student_id', 'id'], 'Student_ID')
mapped_name = find_dynamic_column(['name', 'nama'], 'Name')
mapped_gender = find_dynamic_column(['gender', 'jantina', 'sex'], 'Gender')
mapped_dept = find_dynamic_column(['department', 'dept', 'faculty'], 'Department')
mapped_text = find_dynamic_column(['openended', 'confession', 'text', 'answer'], 'Confession_Text')
mapped_status = find_dynamic_column(['status', 'level', 'depression_level'], 'DASS_Depression_Level')
mapped_dass_score = find_dynamic_column(['score', 'dass_score', 'total', 'Score'], 'DASS_Depression_Score')

# Validate that essential components exist
if not mapped_text:
    raise KeyError("Critical Error: Could not dynamically identify any open-ended or text narrative columns.")

print("\n--- Mapped Data Schema Resolution ---")
print(f" Detected ID Column         ->  {mapped_id}")
print(f" Detected Text Column       ->  {mapped_text}")
print(f" Detected Ground Truth Label ->  {mapped_status if mapped_status else 'None (Unlabeled Operational Data)'}")

# Standardize DataFrame columns dynamically
df = pd.DataFrame()
if mapped_id: df['Student_ID'] = raw_df[mapped_id]
if mapped_name: df['Name'] = raw_df[mapped_name]
if mapped_gender: df['Gender'] = raw_df[mapped_gender]
if mapped_dept: df['Department'] = raw_df[mapped_dept]
df['Confession_Text'] = raw_df[mapped_text]
if mapped_status: df['DASS_Depression_Level'] = raw_df[mapped_status]


# =====================================================================
# 2. RUNTIME PRE-PROCESSING & FEATURE EXTRACTOR
# =====================================================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(processed_words)

print("\nExecuting Preprocessing Engine...")
df['Preprocessed_Text'] = df['Confession_Text'].apply(preprocess_text)

print("Calculating VADER Polarity Feature...")
analyzer = SentimentIntensityAnalyzer()
df['Lexicon_Compound_Score'] = df['Confession_Text'].apply(lambda txt: analyzer.polarity_scores(str(txt))['compound'])

def interpret_prediction_to_percentage(probabilities, classes):
    """
    Maps multi-class prediction probabilities to your specific DASS clinical percentage scale.
    """
    bracket_midpoints = {
        'Normal': 11.5,
        'Mild': 27.95,
        'Moderate': 41.0,
        'Severe': 56.9,
        'Extremely Severe': 82.15
    }
    
    weighted_score = 0.0
    prob_dict = dict(zip(classes, probabilities))
    
    for label, prob in prob_dict.items():
        weighted_score += prob * bracket_midpoints[label]
        
    if weighted_score <= 23.0:
        assigned_level = "Normal"
    elif weighted_score <= 32.0:
        assigned_level = "Mild"
    elif weighted_score <= 50.0:
        assigned_level = "Moderate"
    elif weighted_score <= 64.0:
        assigned_level = "Severe"
    else:
        assigned_level = "Extremely Severe"
        
    return weighted_score, assigned_level

# =====================================================================
# 3. LOAD PERSISTED PIPELINE ASSETS & PREDICT WITH CLINICAL PROBABILITY
# =====================================================================
print("\nLoading pre-trained model files...")
try:
    tfidf_vectorizer = joblib.load(r"C:\Users\Izzah\Documents\3. FYP\src\tfidf_vectorizer.pkl")
    trained_model = joblib.load(r"C:\Users\Izzah\Documents\3. FYP\output\hybrid_engine.pkl")
except FileNotFoundError as e:
    print(f"Pipeline Load Error: Ensure you have exported your trained model components. {e}")
    exit()

# Transform validation text using the same trained NLP
tfidf_matrix = tfidf_vectorizer.transform(df['Preprocessed_Text']).toarray()
tfidf_cols = [r'TFIDF_' + word for word in tfidf_vectorizer.get_feature_names_out()]
df_tfidf = pd.DataFrame(tfidf_matrix, columns=tfidf_cols)

# Dynamically extract the real score if found, or map it properly
# If Solution A from earlier used raw_df, this one make sure it's injected right here into 'df'
if 'mapped_dass_score' in locals() and mapped_dass_score:
    df['DASS_Depression_Score'] = raw_df[mapped_dass_score]
else:
    # Fallback to safe neutral baseline placeholder when the sheet only had status and text strings
    df['DASS_Depression_Score'] = raw_df[find_dynamic_column(['score', 'dass_score', 'total', 'dass'], 'DASS_Depression_Score')] if find_dynamic_column(['score', 'dass_score', 'total', 'dass'], 'DASS_Depression_Score') else 0

# Reconstruct exact clinical matching matrix layout expected by model
X_validation = pd.concat([
    df[['DASS_Depression_Score', 'Lexicon_Compound_Score']], 
    df_tfidf
], axis=1)

print("Routing validation features through classification engine...")

# Extract the raw multi-class probabilities instead of just hard predictions levels
probabilities = trained_model.predict_proba(X_validation)

# Pass probabilities through DASS interpretation calculation engine loop
leaning_percentages = []
interpreted_levels = []

for i in range(len(df)):
    pct, lvl = interpret_prediction_to_percentage(probabilities[i], trained_model.classes_)
    leaning_percentages.append(round(pct, 2))  # Keep it neat to 2 decimal places
    interpreted_levels.append(lvl)

# Map the calculated lists into your tracking DataFrame columns
df['Depression_Leaning_Percentage'] = leaning_percentages
df['Model_Predicted_Level'] = interpreted_levels

print("[PREDICTION] Successfully extracted continuous probability leanings and assigned levels.")

# =====================================================================
# 4. PERFORMANCE ACCURACY EVALUATION 
# =====================================================================
if 'DASS_Depression_Level' in df.columns:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Calculate overall accuracy
    val_acc = accuracy_score(df['DASS_Depression_Level'], df['Model_Predicted_Level']) * 100
    
    # Extract averaged precision, recall, and f1-score across all 5 classes
    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
        df['DASS_Depression_Level'], 
        df['Model_Predicted_Level'], 
        average='macro', 
        zero_division=0
    )
    
    print("\n" + "="*65)
    print("             BLIND VALIDATION PERFORMANCE MATRIX             ")
    print("="*65)
    print(f"   Accuracy  : {val_acc:.2f}%")
    print(f"   Precision : {val_prec * 100:.2f}% (Minimizes False Alarms)")
    print(f"   Recall    : {val_rec * 100:.2f}% (Clinical Sensitivity - Safety Metric)")
    print(f"   F1-Score  : {val_f1 * 100:.2f}% (Balanced Performance Harmonization)")
    print("="*65)
    print("💡 Note: These metrics prove how robustly the model generalizes")
    print("   to completely fresh dataset files handed over by the panel.")
    print("="*65)
else:
    print("\n[VALIDATION NOTE] No baseline target labels found. Operating in raw predictive inference mode.")

# =====================================================================
# 5. POWER BI COMPATIBLE EXPORT
# =====================================================================
export_output_path = r"C:\Users\Izzah\Documents\3. FYP\output\Validation_Predictions_PowerBI.csv"
df.to_csv(export_output_path, index=False)
print(f"\n[EXPORT] Clean predictive tracking matrix saved to: {export_output_path}")

# =====================================================================
# 6. POWER BI PERFORMANCE METRICS EXPORT & VISUALIZATION
# =====================================================================
if 'DASS_Depression_Level' in df.columns:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # 1. Calculate overall multi-class metrics
    val_acc = accuracy_score(df['DASS_Depression_Level'], df['Model_Predicted_Level'])
    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
        df['DASS_Depression_Level'], 
        df['Model_Predicted_Level'], 
        average='macro', 
        zero_division=0
    )
    
    # 2. Structure them into a clean, flat dataframe
    metrics_data = {
        'Metric_Name': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Metric_Value': [round(val_acc, 4), round(val_prec, 4), round(val_rec, 4), round(val_f1, 4)],
        'Description': [
            'Overall percentage of exact classification matches.',
            'Macro precision (minimizes false alarms/wrong flags).',
            'Macro recall (clinical sensitivity - capturing actual cases).',
            'Harmonic mean balancing precision and recall.'
        ]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Save CSV for Power BI
    metrics_output_path = r"C:\Users\Izzah\Documents\3. FYP\output\Validation_Metrics_PowerBI.csv"
    df_metrics.to_csv(metrics_output_path, index=False)
    print(f"[EXPORT] Validation Metrics CSV saved to: {metrics_output_path}")

    # =====================================================================
    # VISUALIZATION GENERATION 
    # =====================================================================
    # Convert values to percentages for clear display 
    plot_df = df_metrics.copy()
    plot_df['Percentage'] = plot_df['Metric_Value'] * 100

    # Set up styling parameters matching the UTP Corporate scheme
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Create the horizontal bar plot
    # Using a professional deep navy palette color
    ax = sns.barplot(
        x='Percentage', 
        y='Metric_Name', 
        data=plot_df, 
        palette=['#003366', '#004080', '#0059b3', '#0073e6'],
        hue='Metric_Name',
        legend=False
    )
    
    # Customise titles and labels
    plt.title('Blind Validation Performance Metrics Evaluation Profile', fontsize=14, fontweight='bold', pad=15, color='#003366')
    plt.xlabel('Score Percentage (%)', fontsize=12, fontweight='semibold')
    plt.ylabel('Evaluation Parameter', fontsize=12, fontweight='semibold')
    plt.xlim(0, 105) # Add padding room for value labels
    
    # Add numerical labels directly onto the ends of the bars
    for index, row in plot_df.iterrows():
        ax.text(
            row['Percentage'] + 1, 
            index, 
            f"{row['Percentage']:.2f}%", 
            color='black', 
            ha="left", 
            va="center", 
            fontweight='bold',
            fontsize=11
        )
        
    plt.tight_layout()
    
    # Define image path and save the file
    visual_output_path = r"C:\Users\Izzah\Documents\3. FYP\output\validation_performance_metrics.png"
    plt.savefig(visual_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[IMAGE] Performance evaluation chart exported successfully!")
    print(f"        Destination: {visual_output_path}")
    print("="*65)
else:
    print("\n[EXPORT SKIPPED] Cannot compute metrics because baseline target labels are missing.")