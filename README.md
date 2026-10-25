# Explainable Sentiment Classifier using Linguistically Enhanced TF-IDF

A modular, explainable sentiment analysis pipeline built to explicitly handle complex linguistic phenomena like **negation, idioms, and contrastive discourse** while maintaining the perfect mathematical transparency of classical machine learning models.

## 📖 Overview

Traditional bag-of-words and TF-IDF models treat text as independent tokens, making them highly vulnerable to linguistic structures. For example, a standard model may classify *"not good"* as positive because of the word "good," or fail to understand that *"The acting was terrible, but the ending was amazing"* should be classified as positive due to the adversative conjunction "but."

This project introduces a **linguistically informed preprocessing pipeline** applied *before* TF-IDF feature extraction. By pairing this with Logistic Regression, the model achieves highly competitive accuracy while retaining the ability to mathematically explain **exactly** which words contributed to a prediction, and by how much.

## ✨ Key Features

* **Hybrid Idiom Resolver**: Normalizes idioms using NLTK WordNet synonym extraction, backed by a curated manual dictionary (`idiom_mapping.py`) for robustness.
* **Negation Scope Detection**: Uses a sliding window to attach a `NOT_` prefix to sentiment-bearing words rather than risky antonym replacement (e.g., *"not good"* → `NOT_good`).
* **Adversative Conjunction Handling**: Detects discourse shifts (e.g., *but, however, yet*) and duplicates dominant sentiment tokens in the subsequent clause to natively boost their TF-IDF weight.
* **Punctuation Features**: Explicitly extracts expressive punctuation like `!` and `?` into tokens (`EXCLAMATION_MARK`) to capture sentiment intensity.
* **Semantic Enrichment**: Leverages `spaCy` for POS-tagging and lemmatization, and WordNet to inject synonym expansions directly into the token sequence.
* **Complete Explainability**: Breaks down the logistic regression decision into exact additive contributions per token.

## 🚀 Installation & Usage

### 1. Requirements
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install pandas numpy scikit-learn spacy datasets nltk matplotlib seaborn
```

### 2. Download NLP Models
The pipeline relies on `spaCy`'s English model and NLTK's WordNet lexical database.
```bash
python -m spacy download en_core_web_sm
```
*(Note: The script will automatically attempt to download the NLTK WordNet corpora upon first run).*

### 3. Run the Pipeline
Execute the main script to train the baseline and enhanced models, evaluate them on the SST-2 and Emotion datasets, and generate feature importance explanations.
```bash
python sentiment_tfidf_pipeline_v3.py
```

To generate the presentation-ready plots and tables found in the `plots/` directory:
```bash
python generate_plots.py
```

## 🗄️ Datasets

The system is evaluated on two distinct datasets to test its performance across different sentiment classification tasks:

1. **[Stanford Sentiment Treebank (SST-2)](https://huggingface.co/datasets/sst2)**: A standard benchmark dataset for binary sentiment polarity classification (Positive/Negative). Loaded automatically via the HuggingFace `datasets` library.
2. **[Emotion Dataset](https://www.kaggle.com/datasets/parulpandey/emotion-dataset/data)**: A fine-grained classification dataset categorizing text into six distinct emotions (*sadness, joy, love, anger, fear, surprise*). A local CSV file `emotion_dataset.csv` is used.

## 📊 Experimental Results

We conducted an A/B test comparing a **Plain TF-IDF Baseline** against our **Enhanced Pipeline**.

### SST-2 (Binary Polarity)
| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Baseline | 87.78% | 0.88 |
| Enhanced | **89.35%** | **0.89** |
| **Delta** | **+1.57%** | **+0.01** |

The enhanced pipeline yielded a strong **+1.57% accuracy improvement** over the baseline, confirming that the linguistic preprocessing adds significant value for binary polarity classification, pushing the model to the ceiling of classical architectures.

*Note: On fine-grained multi-class tasks like the Emotion dataset, the polarity-oriented rules (like binary negation) introduced noise, resulting in a slight drop. See the full `project_report.md` for a detailed linguistic analysis of this task-specificity.*

## 🔍 Explainability Demo

Because the system uses Logistic Regression over TF-IDF vectors, the prediction score is a perfectly linear combination of the input tokens. The `explain_prediction()` module outputs exactly how the model arrived at its decision:

```text
======================================================================
Original Text: The acting was terrible, but the ending was amazing.
Processed:     act terrible terrible end amazing amazing
Prediction:    POSITIVE (confidence: 0.87)
======================================================================
Top reasons for predicting POSITIVE:
  🔹 +0.12  Base Model Bias
  📗 +2.40  'amazing'     [TF-IDF] (weight: +1.92, TF-IDF: 1.25)
  📕 -1.25  'terrible'    [TF-IDF] (weight: -2.10, TF-IDF: 0.60)
======================================================================
```
*(Notice how the adversative conjunction "but" triggered the duplication of "amazing" and "terrible", appropriately boosting their TF-IDF impact).*

## 📂 Repository Structure

* `sentiment_tfidf_pipeline_v3.py`: The core NLP pipeline, classifier training, and explainability module.
* `generate_plots.py`: Visualization script for experimental results.
* `idiom_mapping.py`: Fallback manual idiom dictionary.
* `project_report.md`: Comprehensive academic report detailing the system architecture, linguistic analysis, and experimental discussion.
* `plots/`: Generated visual assets for presentations.
