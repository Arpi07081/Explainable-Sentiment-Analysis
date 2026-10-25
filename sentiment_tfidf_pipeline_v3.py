# This script removed static embeddings from v2.

import pandas as pd
import numpy as np
import re
import spacy
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from datasets import load_dataset
import joblib
import os
import nltk
from nltk.corpus import wordnet as wn

# Ensure WordNet is downloaded
try:
    wn.synsets('dog')
except LookupError:
    print("Downloading NLTK WordNet...")
    nltk.download('wordnet', quiet=True)

# Import the custom idiom dictionary you created
from idiom_mapping import IDIOM_MAP

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

NEGATION_WORDS = {
    "not", "no", "never", "n't", "neither", "nor", "nobody", 
    "nothing", "nowhere", "barely", "hardly", "scarcely"
}

ADV_CONJUNCTIONS = {"but", "however", "yet", "although"}

SORTED_IDIOMS = sorted(IDIOM_MAP.keys(), key=lambda x: len(x.split()), reverse=True)

# -------------------------------------------------------------------
# Hybrid WordNet Idiom Precomputation
# -------------------------------------------------------------------
HYBRID_IDIOM_MAP = {}
for idiom in SORTED_IDIOMS:
    wn_idiom = idiom.replace(' ', '_')
    synsets = wn.synsets(wn_idiom)
    
    replacement = None
    if synsets:
        for synset in synsets:
            for lemma in synset.lemma_names():
                if '_' not in lemma and lemma.lower() != wn_idiom.lower():
                    replacement = lemma.lower()
                    break
            if replacement:
                break
                
    if not replacement:
        replacement = IDIOM_MAP[idiom]
        
    HYBRID_IDIOM_MAP[idiom] = replacement

# ISSUE 3 FIX: Use 'en_core_web_sm' for speed
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    print("Downloading spaCy 'en_core_web_sm' model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')


# -------------------------------------------------------------------
# Preprocessing Modules
# -------------------------------------------------------------------

def replace_idioms(text: str) -> str:
    """Replaces idiomatic expressions using the precomputed Hybrid WordNet map."""
    for idiom in SORTED_IDIOMS:
        replacement = HYBRID_IDIOM_MAP[idiom]
        pattern = re.compile(r'\b' + re.escape(idiom) + r'\b', re.IGNORECASE)
        text = pattern.sub(replacement, text)
    return text


_antonym_cache = {}

def get_antonym(word):
    if word in _antonym_cache:
        return _antonym_cache[word]
    antonyms = []
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    result = antonyms[0].replace('_', ' ') if antonyms else None
    _antonym_cache[word] = result
    return result


def expand_with_synonyms(lemma: str, pos_tag: str, max_synonyms: int = 1) -> list:
    """Returns the original lemma plus its top WordNet synonym(s)."""
    # Map spaCy POS tags to WordNet POS
    pos_map = {'NOUN': wn.NOUN, 'VERB': wn.VERB, 'ADJ': wn.ADJ, 'ADV': wn.ADV}
    wn_pos = pos_map.get(pos_tag)
    
    synonyms = []
    synsets = wn.synsets(lemma, pos=wn_pos) if wn_pos else wn.synsets(lemma)
    for syn in synsets[:1]:  # Only top synset for precision
        for syn_lemma in syn.lemma_names()[:max_synonyms]:
            candidate = syn_lemma.replace('_', ' ').lower()
            if candidate != lemma and '_' not in syn_lemma:
                synonyms.append(candidate)
    return synonyms


def preprocess_text(text: str, enhanced: bool = True) -> str:
    """
    Pipeline:
    1. Idiom replacement
    2. spaCy Tokenization & POS Tagging
    3. Window-based Negation Handling
    4. POS-based Adversative Duplication
    5. Punctuation Feature Extraction (!, ?)
    """
    if enhanced:
        text = replace_idioms(text)
    
    doc = nlp(text)
    tokens = []
    
    negation_window = 0
    adversative_active = False
    
    for token in doc:
        if enhanced and token.lower_ in ADV_CONJUNCTIONS:
            adversative_active = True
            
        if enhanced and token.lower_ in NEGATION_WORDS:
            negation_window = 3
            continue 
            
        if token.is_punct:
            if enhanced and token.text == "!":
                tokens.append("EXCLAMATION_MARK")
            elif enhanced and token.text == "?":
                tokens.append("QUESTION_MARK")
                
            negation_window = 0
            continue
            
        if token.is_space:
            continue
            
        if token.is_stop and negation_window == 0:
            continue
            
        lemma = token.lemma_.lower()
        
        if enhanced and negation_window > 0:
            antonym = get_antonym(lemma)
            if antonym:
                lemma = antonym
                negation_window = 0
            else:
                lemma = "NOT_" + lemma
                negation_window -= 1
            
        tokens.append(lemma)
        if enhanced and negation_window == 0 and token.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV'):
            tokens.extend(expand_with_synonyms(lemma, token.pos_))
        
        if enhanced and adversative_active and token.pos_ in ('ADJ', 'ADV', 'VERB'):
            tokens.append(lemma)
            adversative_active = False
        
    return " ".join(tokens)


# -------------------------------------------------------------------
# Explainability Modules
# -------------------------------------------------------------------

def print_global_feature_importance(vectorizer: TfidfVectorizer, model: LogisticRegression, class_names: list, top_n: int = 10):
    """Prints the most globally important features learned by the model for each class."""
    feature_names = vectorizer.get_feature_names_out()
    
    print("=" * 60)
    print(" 🌟 GLOBAL FEATURE IMPORTANCE (LR Coefficients)")
    print("=" * 60)
    
    if len(class_names) == 2:
        coef = model.coef_[0]
        sorted_indices = np.argsort(coef)
        
        print(f"\nTop Features for {class_names[1].upper()}:")
        print(f"{'Feature':<25} | {'Weight':>8}")
        print("-" * 36)
        for idx in sorted_indices[-top_n:][::-1]:
            print(f"{feature_names[idx]:<25} | {coef[idx]:>+8.4f}")
            
        print(f"\nTop Features for {class_names[0].upper()}:")
        print(f"{'Feature':<25} | {'Weight':>8}")
        print("-" * 36)
        for idx in sorted_indices[:top_n]:
            print(f"{feature_names[idx]:<25} | {coef[idx]:>+8.4f}")
    else:
        for class_idx, class_name in enumerate(class_names):
            coef = model.coef_[class_idx]
            sorted_indices = np.argsort(coef)
            print(f"\nTop Features for {class_name.upper()}:")
            print(f"{'Feature':<25} | {'Weight':>8}")
            print("-" * 36)
            for idx in sorted_indices[-top_n:][::-1]:
                print(f"{feature_names[idx]:<25} | {coef[idx]:>+8.4f}")
                
    print("=" * 60 + "\n")


def explain_prediction(text: str, vectorizer: TfidfVectorizer, model: LogisticRegression, enhanced: bool, class_names: list):
    """Decomposes the Logistic Regression prediction for a specific sentence."""
    processed = preprocess_text(text, enhanced=enhanced)
    vec = vectorizer.transform([processed])
    pred_idx = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    
    feature_names = vectorizer.get_feature_names_out()
        
    # Handle binary vs multiclass prediction weights
    if len(class_names) == 2:
        coef = model.coef_[0]
        intercept = model.intercept_[0]
    else:
        coef = model.coef_[pred_idx]
        intercept = model.intercept_[pred_idx]
        
    nonzero_indices = vec.nonzero()[1]
    
    contributions = []
    for idx in nonzero_indices:
        feat_name = feature_names[idx]
        tf_idf_val = vec[0, idx]
        weight = coef[idx]
        contribution = tf_idf_val * weight
        contributions.append((feat_name, tf_idf_val, weight, contribution))
        
    contributions.sort(key=lambda x: abs(x[3]), reverse=True)
    
    print("=" * 70)
    print(f"Original Text: {text}")
    print(f"Processed:     {processed}")
    print(f"Prediction:    {class_names[pred_idx].upper()} (confidence: {max(prob):.2f})")
    print("=" * 70)
    
    if len(class_names) == 2:
        print(f"Top reasons for predicting {class_names[1].upper()} (negative values pull towards {class_names[0].upper()}):")
    else:
        print(f"Top reasons for predicting {class_names[pred_idx].upper()}:")
    
    print(f"  {'🔹' if intercept > 0 else '🔻'} {intercept:+5.2f}  Base Model Bias")
    
    for feat_name, tf_idf_val, weight, contribution in contributions[:15]:
        icon = "📗" if contribution > 0 else "📕"
        print(f"  {icon} {contribution:+5.2f}  '{feat_name}' (weight: {weight:+.2f}, TF-IDF: {tf_idf_val:.2f})")
    print("=" * 70)
    print()


# -------------------------------------------------------------------
# Inference Module
# -------------------------------------------------------------------

def predict_new_sentences(sentences: list, vectorizer_path: str = "sst-2_tfidf_vectorizer.pkl", 
                          model_path: str = "sst-2_logistic_regression_model.pkl", 
                          enhanced: bool = True, class_names: list = ['Negative', 'Positive']):
    """
    Loads saved models from disk and predicts sentiment for a list of new sentences.
    Provides detailed explainability for each prediction.
    """
    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        print(f"❌ Error: Model files not found. Ensure '{vectorizer_path}' and '{model_path}' exist.")
        print("Please run the main() training pipeline first.")
        return

    print(f"\nLoading saved models from '{vectorizer_path}' and '{model_path}'...")
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    print("✅ Models loaded successfully.")
    
    print("\n" + "#"*50)
    print(" 🔮 INFERENCE ON NEW SENTENCES")
    print("#"*50 + "\n")
    
    for sentence in sentences:
        explain_prediction(sentence, vectorizer, model, enhanced=enhanced, class_names=class_names)


# -------------------------------------------------------------------
# Main Execution Pipeline
# -------------------------------------------------------------------

def train_and_evaluate(df, enhanced: bool, class_names: list):
    """Trains and evaluates the pipeline, allowing for A/B testing."""
    print(f"\n[{'ENHANCED' if enhanced else 'BASELINE'}] Running Preprocessing...")
    tqdm.pandas()
    processed_texts = df['sentence'].progress_apply(lambda x: preprocess_text(x, enhanced=enhanced))
    
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        processed_texts, df['sentiment'], 
        test_size=0.15, random_state=42, stratify=df['sentiment']
    )
    
    print(f"[{'ENHANCED' if enhanced else 'BASELINE'}] Vectorizing TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=15000, 
        ngram_range=(1, 2), 
        sublinear_tf=True, 
        min_df=2
    )
    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)
    
    print(f"[{'ENHANCED' if enhanced else 'BASELINE'}] Training Logistic Regression...")
    clf = LogisticRegression(C=5.0, max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n--- {'ENHANCED' if enhanced else 'BASELINE'} RESULTS ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return vectorizer, clf, acc


def run_pipeline_for_dataset(df, dataset_name, file_prefix, class_names, test_sentences):
    print("\n\n" + "!"*70)
    print(f"🚀 STARTING PIPELINE FOR DATASET: {dataset_name}")
    print("!"*70)
    
    # Model A: Baseline
    print("\n" + "*"*50)
    print(f" MODEL A: PLAIN TF-IDF BASELINE ({dataset_name})")
    print("*"*50)
    tfidf_base, clf_base, acc_base = train_and_evaluate(df, enhanced=False, class_names=class_names)
    
    # Model B: Enhanced Pipeline
    print("\n" + "*"*50)
    print(f" MODEL B: ENHANCED LINGUISTIC PIPELINE ({dataset_name})")
    print("*"*50)
    tfidf_enh, clf_enh, acc_enh = train_and_evaluate(df, enhanced=True, class_names=class_names)
    
    # Summary
    print("\n" + "="*50)
    print(f" 🏆 EXPERIMENT RESULTS SUMMARY ({dataset_name})")
    print("="*50)
    print(f"Model A (Baseline) Accuracy: {acc_base:.4f}")
    print(f"Model B (Enhanced) Accuracy: {acc_enh:.4f}")
    print(f"Improvement:                 {(acc_enh - acc_base)*100:+.2f}%\n")
    
    # Feature Importance
    print_global_feature_importance(tfidf_enh, clf_enh, class_names=class_names, top_n=10)
    
    # Save Models
    print("\n" + "="*50)
    print(f" 💾 SAVING TRAINED MODELS ({dataset_name})")
    print("="*50)
    
    joblib.dump(tfidf_enh, f"{file_prefix}_tfidf_vectorizer.pkl")
    joblib.dump(clf_enh, f"{file_prefix}_logistic_regression_model.pkl")
    print(f"✅ Saved '{file_prefix}_tfidf_vectorizer.pkl'")
    print(f"✅ Saved '{file_prefix}_logistic_regression_model.pkl'")
    
    # Explainability Demo
    print("\n" + "#"*50)
    print(f" 🔍 EXPLAINABILITY DEMONSTRATION ({dataset_name} Enhanced Model)")
    print("#"*50 + "\n")
    for sentence in test_sentences:
        explain_prediction(sentence, tfidf_enh, clf_enh, enhanced=True, class_names=class_names)


def main():
    # ⚠️ SUBSAMPLING FOR SPEED
    SUBSAMPLE_N = 5000 
    
    # ==========================================
    # DATASET 1: SST-2 (Binary Sentiment)
    # ==========================================
    print("Loading SST-2 dataset...")
    sst2 = load_dataset('glue', 'sst2')
    train_df = sst2['train'].to_pandas()[['sentence', 'label']]
    val_df = sst2['validation'].to_pandas()[['sentence', 'label']]
    df_sst2 = pd.concat([train_df, val_df], ignore_index=True)
    df_sst2.rename(columns={'label': 'sentiment'}, inplace=True)
    
    if SUBSAMPLE_N:
        print(f"\nSubsampling {SUBSAMPLE_N} rows for fast testing on SST-2...")
        df_sst2 = df_sst2.sample(n=SUBSAMPLE_N, random_state=42).reset_index(drop=True)
    else:
        print(f"\nRunning on full SST-2 dataset: {len(df_sst2)} rows.")
        
    sst2_test_sentences = [
        "The movie was not good at all.",
        "I do not like this movie because the acting was boring.",
        "It was a piece of cake to understand the plot.",
        "The acting was terrible, but the visual effects were amazing.",
        "This film is a blessing in disguise for the sci-fi genre."
    ]
    
    run_pipeline_for_dataset(
        df_sst2, 
        dataset_name="SST-2 (Binary Sentiment)",
        file_prefix="sst-2",
        class_names=['Negative', 'Positive'], 
        test_sentences=sst2_test_sentences
    )
    
    # ==========================================
    # DATASET 2: Emotion Dataset (Multiclass)
    # ==========================================
    print("\nLoading Emotion Dataset...")
    if os.path.exists("emotion_dataset.csv"):
        df_emotion = pd.read_csv("emotion_dataset.csv")
        # Ensure correct column names
        df_emotion.rename(columns={'text': 'sentence', 'label': 'sentiment'}, inplace=True)
        
        if SUBSAMPLE_N and len(df_emotion) > SUBSAMPLE_N:
            print(f"\nSubsampling {SUBSAMPLE_N} rows for fast testing on Emotion Dataset...")
            df_emotion = df_emotion.sample(n=SUBSAMPLE_N, random_state=42).reset_index(drop=True)
        else:
            print(f"\nRunning on full Emotion Dataset: {len(df_emotion)} rows.")
            
        emotion_class_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        
        emotion_test_sentences = [
            "I am absolutely furious about this terrible service!",
            "I feel so completely heartbroken and lost without you.",
            "This is the most wonderful surprise I have ever received!",
            "I am terrified of the dark and alone in this house.",
            "I really love spending time with my amazing friends."
        ]
        
        run_pipeline_for_dataset(
            df_emotion, 
            dataset_name="Emotion Dataset (6-Class)", 
            file_prefix="emotion",
            class_names=emotion_class_names, 
            test_sentences=emotion_test_sentences
        )
    else:
        print("❌ 'emotion_dataset.csv' not found in the current directory. Skipping Emotion Dataset.")

if __name__ == "__main__":
    main()
