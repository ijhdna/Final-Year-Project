import pandas as pd
import numpy as np
import random

# =====================================================================
# 1. INITIALIZATION AND BASE DATA SEEDING
# =====================================================================
np.random.seed(42)
random.seed(42)

# Load the example dummy file
input_file = r"C:\Users\Izzah\Documents\3. FYP\src\DASS_Dummy.csv"
try:
    df_example = pd.read_csv(input_file)
    print("[SETUP] Loaded example reference dataset successfully.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Extract categorical choices directly from your example data to maintain realism
departments = list(df_example['Department'].unique())
genders = list(df_example['Gender'].unique())
semesters = list(df_example['Semester'].unique())

# Strict 5-class score brackets based on your DASS-21 direct rules
status_brackets = {
    'Normal': (0, 4), 
    'Mild': (5, 6), 
    'Moderate': (7, 10), 
    'Severe': (11, 13), 
    'Extremely Severe': (14, 21)
}
status_options = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']

# =====================================================================
# 2. ISOLATING VOCABULARY POOLS BY CLINICAL SEVERITY
# =====================================================================
text_pools = {status: [] for status in status_options}

# Populate text pools dynamically from your example data to ensure vocabulary alignment
for status in status_options:
    matching_texts = df_example[df_example['DASS_Depression_Level'] == status]['Confession_Text'].tolist()
    text_pools[status].extend(matching_texts)

# Add fallback diverse phrases to enrich the pools and completely prevent exact row duplication
extra_phrases = {
    'Normal': [
        "Everything is running smoothly this week.", "Classes are manageable if I pace myself well.",
        "Feeling steady and optimistic about the remaining weeks of the semester.", "Enjoying campus life right now.",
        "I catch up on rest during the weekends.", "I feel like I have a solid grasp on my current subjects.",
        "I still look forward to my favorite meals and streaming my favorite shows."
    ],
    'Mild': [
        "A bit anxious about my upcoming project presentation milestones.", "Group assignments are moving a bit slow.",
        "Minor overthinking about my current grades, hopefully things get better soon.", "A little bit tired.",
        "My shoulders have been feeling really tight and tense during lectures this past week.",
        "I’ve been a bit more impatient than usual when people walk too slowly in front of me.",
        "I keep worrying about whether I'll find a good internship placement for next semester, and it's lingering in the back of my mind."
    ],
    'Moderate': [
        "Lately I find it really hard to concentrate during long group project meetings.", "My chest feels tight occasionally.",
        "Struggling to find the energy to finish my lab reports on time.", "Quite anxious and lonely lately.",
        "I’m losing my temper over tiny things.", "I get deeply upset whenever anything delays my schedule by even ten minutes.",
        "I can force myself to go to classes, but I feel like an absolute robot."
    ],
    'Severe': [
        "I feel a profound sense of sadness every morning when I open my eyes.", "I have started avoiding my friends.",
        "Everything feels useless and dark, the academic pressure is setting in.", "Drowning in stress right now.",
        "I haven’t attended a single lecture this week because the thought of walking into the hall and having people look at me makes me sick.",
        "I am so incredibly stressed that my stomach has been in painful knots for days, and I can barely swallow food.",
        "Everything feels like a massive chore; even showering or taking out the trash takes every single ounce of energy I have left."
    ],
    'Extremely Severe': [
        "I feel completely broken and helpless against this constant wave of deep exhaustion.", "No matter how much I sleep, I wake up feeling miserable.",
        "Trapped in total darkness and feeling completely empty every single day.", "I do not have the energy to talk to anyone.", 
        "I feel like I am under an unbearable amount of pressure that never, ever lets up for a single second, and I just want to disappear.",
        "I’ve completely stopped eating and sleeping properly because my mind is in a non-stop, agonizing loop of stress and exhaustion.",
        "I’ve missed my assignment deadlines, but I can’t even bring myself to email my lecturer."
    ]
}

for status in status_options:
    text_pools[status].extend(extra_phrases[status])

# =====================================================================
# 3. SYNTHETIC GENERATION ENGINE
# =====================================================================
synthetic_rows = []
total_rows_to_generate = 200

for i in range(total_rows_to_generate):
    # Select a target clinical status class
    assigned_status = random.choice(status_options)
    
    # 1. Generate realistic UTP Student ID (first 2 digits between 21 and 26)
    year_prefix = random.choice([21, 22, 23, 24, 25, 26])
    random_digits = random.randint(100000, 999999)
    student_id = int(f"{year_prefix}{random_digits}")
    
    # 2. Permute metadata features from the example data distributions
    gender = random.choice(genders)
    dept = random.choice(departments)
    semester = random.choice(semesters)
    
    # 3. Force numerical score to bind tightly within its clinical class boundary
    score_range = status_brackets[assigned_status]
    final_score = random.randint(score_range[0], score_range[1])
    
    # 4. Construct unique text profiles by sampling and combining sentence elements
    pool = text_pools[assigned_status]
    sentence_count = random.choice([1, 2])
    chosen_sentences = random.sample(pool, min(sentence_count, len(pool)))
    confession_text = " ".join(chosen_sentences)
    
    # Append the completed synthetic row
    synthetic_rows.append({
        'Student_ID': student_id,
        'Gender': gender,
        'Department': dept,
        'Semester': semester,
        'DASS_Depression_Score': final_score,
        'Confession_Text': confession_text,
        'DASS_Depression_Level': assigned_status
    })

# Convert to DataFrame
df_synthetic = pd.DataFrame(synthetic_rows)

# =====================================================================
# 4. EXPORT STANDALONE DATA ASSET
# =====================================================================
output_path = r"C:\Users\Izzah\Documents\3. FYP\src\Synthetic_Dataset.csv"
df_synthetic.to_csv(output_path, index=False)

print("\n" + "="*60)
print("             SYNTHETIC DATA STEP COMPLETE            ")
print("="*60)
print(f"Standalone Synthetic Rows Generated: {len(df_synthetic)}")
print(f"Exported File Path: {output_path}")
print("\nGenerated Class Distribution Matrix:")
print(df_synthetic['DASS_Depression_Level'].value_counts())
print("="*60)