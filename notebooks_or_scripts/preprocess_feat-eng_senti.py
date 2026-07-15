import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure all necessary NLTK components are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('averaged_perceptron_tagger')      # Keeps older version support
nltk.download('averaged_perceptron_tagger_eng')  # <--- ADD THIS LINE FOR YOUR VERSION
# =====================================================================
# 0. DATA COLLECTION (Load Dataset)
# =====================================================================
try:
    df = pd.read_csv(r'C:\Users\Izzah\Documents\3. FYP\src\Synthetic_Dataset_From_Dummy.csv')
    print("[DATA COLLECTION] Loaded dataset successfully.")
except FileNotFoundError:
    print("Error: Could not find your dataset. Check your file path!")
    exit()

# =====================================================================
# 1. DATA PRE-PROCESSING BLOCK (With POS Tagging for SentiWordNet)
# =====================================================================
print("\n--- Starting Data Pre-Processing ---")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Helper function to convert standard NLTK POS tags to SentiWordNet POS tags
def get_sentiwordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return 'a' # Adjective
    elif nltk_tag.startswith('V'):
        return 'v' # Verb
    elif nltk_tag.startswith('N'):
        return 'n' # Noun
    elif nltk_tag.startswith('R'):
        return 'r' # Adverb
    return None

def preprocess_and_tag(text):
    # Clean text
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    raw_tokens = word_tokenize(text)
    
    # Remove Stopwords first
    meaningful_tokens = [word for word in raw_tokens if word not in stop_words]
    
    # Get POS tags for the words
    tagged_tokens = nltk.pos_tag(meaningful_tokens)
    
    final_tokens = []
    cleaned_words_list = []
    
    for word, tag in tagged_tokens:
        swn_pos = get_sentiwordnet_pos(tag)
        if swn_pos: # Keep only Nouns, Verbs, Adjectives, Adverbs for SentiWordNet
            # Lemmatize using the correct grammatical part of speech
            lemma = lemmatizer.lemmatize(word, pos=swn_pos)
            final_tokens.append((lemma, swn_pos))
            cleaned_words_list.append(lemma)
            
    cleaned_text_str = " ".join(cleaned_words_list)
    return final_tokens, cleaned_text_str

# Apply the updated preprocessing
df['Tagged_Tokens'], df['Preprocessed_Text'] = zip(*df['Confession_Text'].apply(preprocess_and_tag))
print("[PRE-PROCESSING] Cleaning, Lemmatization, and POS Tagging complete.")

# =====================================================================
# 2. FEATURE ENGINEERING BLOCK (SentiWordNet Scoring Engine)
# =====================================================================
print("\n--- Starting Feature Engineering ---")

# 2.2 SentiWordNet Custom Scoring Function
def calculate_swn_sentiment(tagged_tokens):
    sentiment_score = 0.0
    count = 0
    
    for word, pos in tagged_tokens:
        # Find all matching definitions (synsets) in SentiWordNet
        synsets = list(swn.senti_synsets(word, pos))
        
        if synsets:
            # Take the first common definition score
            first_sense = synsets[0]
            # Calculate a local score: Positive score minus Negative score
            word_score = first_sense.pos_score() - first_sense.neg_score()
            sentiment_score += word_score
            count += 1
            
    # Return average sentiment score of the sentence; if no matches, return 0
    return sentiment_score / count if count > 0 else 0.0

# 2.2 SentiWordNet Custom Scoring Function
# (Keep your existing calculate_swn_sentiment function here...)

print("🤖 Running SentiWordNet Lexicon Engine...")
df['Lexicon_Compound_Score'] = df['Tagged_Tokens'].apply(calculate_swn_sentiment)
print("[FEATURE ENGINEERING] SentiWordNet Scoring complete.")

# 2.3 Feature Extraction using Machine Learning (TF-IDF Vectorization)
tfidf = TfidfVectorizer(max_features=10)
tfidf_matrix = tfidf.fit_transform(df['Preprocessed_Text']).toarray()

tfidf_cols = [r'TFIDF_' + word for word in tfidf.get_feature_names_out()]
df_tfidf = pd.DataFrame(tfidf_matrix, columns=tfidf_cols)

# 2.4 Combine EVERYTHING for Power BI Reporting
# We include metadata columns so your final CSV is fully tracebale!
df_final_features = pd.concat([
    df[['Student_ID', 'Gender', 'Department', 'Semester', 'DASS_Depression_Score', 'Lexicon_Compound_Score']], 
    df_tfidf,
    df['DASS_Depression_Level']
], axis=1)

# =====================================================================
# DISPLAY AND EXPORT RESULTS
# =====================================================================
print("\nNew Master Features layout for Model Input (Using SentiWordNet):")
print(df_final_features.head(3).to_string())

export_path = r"C:\Users\Izzah\Documents\3. FYP\src\Engineered_Features_SentiWordNet.csv"
df_final_features.to_csv(export_path, index=False)
print(f"\n[EXPORT] Successfully saved engineered features to: {export_path}")