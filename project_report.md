# Explainable Sentiment Classifier using Linguistically Enhanced TF-IDF Features

## 1. Introduction

Sentiment analysis is a core task in Natural Language Processing (NLP), concerned with automatically identifying the emotional polarity expressed in a piece of text. While classical machine learning approaches — particularly TF-IDF vectorization combined with Logistic Regression — are computationally efficient and inherently interpretable, they rely on a bag-of-words assumption that discards important linguistic structure. This makes them vulnerable to well-known failure cases such as negation (*"not good"* being misclassified as positive) and idiomatic expressions (*"a piece of cake"* being interpreted literally).

The goal of this project is to build an explainable sentiment classifier that addresses these limitations through linguistically informed preprocessing while deliberately preserving the transparency of the underlying model. The system is evaluated on two tasks: binary polarity classification using the **SST-2 dataset**, and fine-grained emotion classification across six categories using a **multi-class emotion dataset**. A systematic A/B comparison between a plain TF-IDF baseline and the linguistically enhanced pipeline is used to empirically validate each design decision.

---

## 2. System Architecture

The full pipeline consists of the following sequential stages:

1. **Hybrid WordNet Idiom Replacement**
2. **Negation Scope Detection and Token Transformation**
3. **Adversative Conjunction Handling**
4. **Punctuation Feature Extraction**
5. **Tokenization, POS Tagging, and Lemmatization** (via spaCy)
6. **Synonym Expansion** (via WordNet)
7. **TF-IDF Vectorization**
8. **Logistic Regression Classification**
9. **Explainability Module**

All linguistic preprocessing steps (1–6) occur before feature extraction, so they directly shape the vocabulary the classifier learns from. This design keeps the model itself simple and fully interpretable while delegating linguistic complexity to the preprocessing layer.

---

## 3. Approach

### 3.1 Idiom Handling
Idiomatic expressions are multi-word units whose meaning cannot be composed from their individual words. A standard TF-IDF model processing *"a piece of cake"* would assign independent weights to "piece" and "cake," neither of which carries the intended meaning of easy. The system addresses this through a **Hybrid Idiom Resolver**:

1. The system first queries WordNet for the idiom as a compound entry (e.g., `piece_of_cake`). If a recognizable single-word synonym is found among its lemmas, that word replaces the idiom.
2. If WordNet does not cover the idiom, a manually curated fallback dictionary (`idiom_mapping.py`) provides a direct mapping.

This two-tier design grounds the system in a recognized lexical resource (WordNet) while ensuring robustness for expressions with poor WordNet coverage.

### 3.2 Negation Handling
Negation fundamentally reverses the sentiment polarity of a word or phrase. Rather than replacing negated words with their antonyms — which can distort meaning, since *"not amazing"* does not straightforwardly mean *"terrible"* — the system applies a prefix transformation:

> *"not good"* → `NOT_good`

A sliding window of three tokens following a negation trigger word (e.g., *not, never, barely, hardly*) is used to define the scope of negation. This allows Logistic Regression to learn the association between negated tokens and their correct polarity directly from training data, without hardcoding semantic assumptions about antonymy.

### 3.3 Adversative Conjunction Handling
Contrastive or adversative conjunctions (e.g., *but, however, although, yet*) signal a discourse shift where the clause following the conjunction typically dominates the overall sentiment. The system detects these conjunctions and duplicates sentiment-bearing tokens (adjectives, adverbs, and verbs) in the following clause:

> *"The acting was terrible, but the ending was amazing."* → `... amazing amazing`

Because TF-IDF scores are directly influenced by term frequency, duplication safely amplifies the weight of the dominant clause's sentiment signal without modifying the classifier or the feature extraction scheme itself.

### 3.4 Punctuation Features
Expressive punctuation carries pragmatic information about sentiment intensity that standard tokenizers discard. The system converts selected punctuation marks into explicit text tokens before vectorization:

* `!` → `EXCLAMATION_MARK`
* `?` → `QUESTION_MARK`

This allows the model to treat emotional emphasis as a learnable feature.

### 3.5 Tokenization, Lemmatization, and Synonym Expansion
The **spaCy** library is used for tokenization, lemmatization, and POS tagging. Lemmatization reduces vocabulary sparsity by collapsing inflected forms to their base form (e.g., *movies, movie, filmed* → *movie, film*). Stopwords are removed unless they appear within a negation window, since negation words are themselves meaningful.

WordNet **synonym expansion** adds the top synonym of content words (nouns, verbs, adjectives, adverbs) to the token sequence. This enriches the feature space with semantically related terms and partially compensates for lexical variation in the test data.

### 3.6 TF-IDF Vectorization
The processed token sequences are converted to numerical vectors using TF-IDF with the following configuration:
* **Unigrams and bigrams** (`ngram_range=(1,2)`): bigrams are critical for capturing negated tokens such as `NOT_good` as single features.
* **Sublinear TF scaling** (`sublinear_tf=True`): replaces raw term frequency with its logarithm, preventing very frequent terms from dominating the feature space disproportionately.
* **Vocabulary cap** (`max_features=15000`) and **minimum document frequency** (`min_df=2`) to filter noise.

### 3.7 Classification Model
Logistic Regression is chosen as the classifier. For a feature vector $x$, it computes:

$$ \hat{y} = \sigma\left(\sum_{i} w_i x_i + b\right) $$

where $x_i$ are TF-IDF feature values and $w_i$ are learned coefficients. The choice is deliberate: the linear scoring function is a prerequisite for the explainability module, since each word's contribution to the final decision is exactly $\text{TF-IDF}(x_i) \times w_i$.

### 3.8 Explainability Module
The system provides two levels of explanation:
1. **Global explanations** display the highest-magnitude model coefficients across the full training corpus, revealing which words are most strongly associated with each sentiment class overall.
2. **Local explanations** decompose a specific prediction by computing each present token's exact contribution:

$$ \text{Contribution}(x_i) = \text{TF-IDF}(x_i) \times w_i $$

The output lists the tokens ranked by the absolute magnitude of their contribution, along with their individual TF-IDF value and learned weight. This makes the reasoning behind every prediction fully transparent and auditable. A sample output for a binary prediction takes the following form:

```text
Prediction: POSITIVE (confidence: 0.87)
  Base Model Bias:  +0.12
  📗 +2.40  'amazing'     (weight: +1.92, TF-IDF: 1.25)
  📕 -1.25  'NOT_good'    (weight: -2.10, TF-IDF: 0.60)
```
This form of explainability is intrinsic to the model rather than post-hoc, meaning it reflects the model's exact decision process rather than an approximation of it.

---

## 4. Linguistic Analysis

### 4.1 Linguistically Relevant Features
Several linguistic phenomena are directly relevant to the sentiment classification task and motivated specific design decisions in this system.

* **Negation** is arguably the most important linguistic feature for polarity classification. In English, negation can be expressed lexically (*not, never, no*), morphologically (*unhappy, dishonest*), and through quantifier words (*barely, hardly, scarcely*). The system targets lexical negation via a fixed-window approach. Morphological negation (e.g., prefixes *un-*, *dis-*) is partially addressed through lemmatization and model training, but is not explicitly handled as a separate rule, which represents a known gap.
* **Discourse structure** plays a critical role in multi-clause sentences. Sentiment expressed in the main or concluding clause tends to dominate over subordinate or contrasted clauses. The adversative conjunction rule targets this phenomenon specifically. From a linguistic standpoint, this reflects the pragmatic principle that the speaker's final or most emphatic assertion carries the primary communicative intent.
* **Lexical semantics** — the relationship between word meanings — is addressed through WordNet integration. Synonymy expansion and idiom normalization both rely on the premise that semantically equivalent expressions should receive similar feature representations. This is linguistically well-motivated but imperfect in practice: synonyms often differ in register, connotation, or degree of intensity (e.g., *good* vs. *excellent*), and treating them as equivalent can introduce noise.
* **Morphology and inflection** are handled through lemmatization. Reducing *running, ran, and runs* to *run* is linguistically sound for sentiment analysis because the sentiment expressed by a verb is typically independent of its tense or aspect.
* **Pragmatic features** such as exclamation marks encode speaker attitude and emotional intensity. From a linguistic perspective, punctuation belongs to the prosodic and pragmatic layers of language, and its inclusion as explicit features is well-justified for sentiment tasks.

### 4.2 Influence of Linguistic Features on Performance
The A/B experimental design allows the contribution of the enhanced pipeline to be isolated from the baseline. 

* The **negation handling mechanism** has the most measurable linguistic impact: without it, words like *good* and *great* would receive positive weights regardless of their negated context, causing systematic misclassification of sentences such as *"I do not like this at all."* The `NOT_` prefix creates a distinct vocabulary entry that the model can associate independently with negative sentiment.
* **Adversative conjunction handling** improves classification on sentences with mixed sentiment, which are common in review data. Without it, the model must rely on the balance of positive and negative words across the whole sentence, which can be unreliable when the two clauses pull in opposite directions.
* **Idiom replacement** benefits precision on a relatively small subset of sentences but avoids potentially severe misclassifications on high-frequency idioms present in training data (e.g., *"break a leg"* being classified as negative due to the word *"break"*).
* **Synonym expansion** increases vocabulary coverage and can help generalize to unseen words, but it introduces risk: if a synonym is ambiguous or belongs to a different sense than intended, it adds misleading features. The design decision to restrict expansion to the top synset and one synonym per word is a deliberate conservative trade-off.

### 4.3 Linguistic Challenges
Several linguistic phenomena remain challenging for this system:
* **Sarcasm and irony** represent the most significant unsolved challenge. Sarcastic utterances like *"Oh, great, another delay"* use positive lexical items to convey negative sentiment. These cases require pragmatic and contextual reasoning that is fundamentally beyond the capacity of bag-of-words models.
* **Intensification and degree** are only partially captured. A model treating *"slightly annoying"* and *"absolutely infuriating"* as equivalent negative instances ignores an important distinction in the strength of sentiment. Degree adverbs (*very, extremely, slightly, barely*) are currently removed as stopwords in the baseline, though the enhanced model retains them within negation windows.
* **Implicit sentiment** — where no explicit opinion word is present but sentiment is clearly implied — is outside the scope of the current approach. For example, *"The film ran for four hours"* implies a negative opinion without any explicitly negative word.
* **Scope ambiguity in negation** presents a hard limit for the window-based approach. In *"I do not think this movie is particularly good"*, the negation technically scopes over an entire embedded clause. A fixed three-token window fails to capture this correctly. Dependency-parse-based negation scope resolution would address this but at significant computational cost.
* **Polysemy** means that the same word may carry different sentiment in different domains. *"Unpredictable"* is negative for a news story but may be positive for a thriller film. WordNet does not resolve this context-dependence, and neither does TF-IDF in general.

---

## 5. Experimental Setup
Two models were compared:
* **Model A (Baseline)**: Plain TF-IDF + Logistic Regression, no linguistic preprocessing.
* **Model B (Enhanced)**: Full linguistically enhanced pipeline + TF-IDF + Logistic Regression.

Both models were evaluated on the same train/test split (85/15, stratified). The SST-2 dataset was used for binary classification and a six-class emotion dataset was used for fine-grained emotion classification. Accuracy and macro-averaged F1 score were used as evaluation metrics.

---

## 6. Results and Discussion

### SST-2 Binary Results — Strong Improvement
| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Baseline | 87.78% | 0.88 |
| Enhanced | 89.35% | 0.89 |
| **Δ (Delta)** | **+1.57%** | **+0.01** |

On the SST-2 dataset, the enhanced pipeline achieved a +1.57% improvement in accuracy, confirming that the linguistic preprocessing adds value for binary polarity classification. The baseline absolute numbers are extremely competitive (87–89%), placing this model at the classical ceiling for TF-IDF + Logistic Regression architectures. Furthermore, the improvement is consistent across both accuracy and F1 score, ruling out class imbalance as a confound.

### Emotion Dataset Results — Task Specificity Drop
| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Baseline | 89.33% | 0.86 |
| Enhanced | 87.75% | 0.84 |
| **Δ (Delta)** | **-1.58%** | **-0.02** |

On the emotion dataset, we observed a -1.58% drop. We attribute this to a task mismatch between the preprocessing design and the nature of fine-grained emotion labels. The negation and adversative rules encode assumptions about a single polarity axis, which does not apply to a six-way emotion space. For example, *"I am not afraid"* processed to `NOT_afraid` is perfectly suitable for binary sentiment but ambiguous for emotion classification (is it relief? confidence?). Additionally, the baseline accuracy of 89.33% indicates that the raw TF-IDF features already possess very strong, unambiguous signals due to the dataset's direct emotional language (e.g., *"I am terrified"*), leaving little room for heuristic rules to improve without adding noise. 

This finding is itself linguistically meaningful: it demonstrates that sentiment polarity and fine-grained emotion are distinct phenomena requiring different feature engineering strategies.

---

## 7. Limitations and Future Work
The primary limitations of the system are those inherent to any rule-based and bag-of-words approach: it cannot model long-range dependencies, contextual polysemy, or pragmatic inference. The negation window is a heuristic approximation of a fundamentally complex linguistic phenomenon. The manual idiom dictionary requires ongoing maintenance and introduces coverage bias.

Future work could address these limitations by: 
1. Incorporating pre-trained static word embeddings (e.g., GloVe) to enrich feature representations with distributional semantic information.
2. Using dependency parsing for more accurate negation scope resolution.
3. Extending the adversative conjunction rule to handle more complex discourse connectives such as *"despite"*, *"even though"*, and *"while"*.
4. Adapting the preprocessing specifically for emotion classification by using resources like the NRC Emotion Lexicon instead of polarity-oriented negation rules.

---

## 8. Conclusion
This project presented a linguistically informed, explainable sentiment classifier built on classical NLP methods. By systematically incorporating idiom handling, negation-aware preprocessing, adversative conjunction weighting, and punctuation feature extraction into a TF-IDF and Logistic Regression pipeline, the system addresses several well-known failure modes of bag-of-words models while preserving strict interpretability. 

The explainability module provides exact, token-level contribution scores that make every prediction transparent. The A/B experimental design validates that linguistic preprocessing yields measurable improvements over the plain baseline for binary sentiment analysis. The linguistic analysis demonstrates that negation, discourse structure, and lexical semantics are the most consequential phenomena for this task, and identifies sarcasm, implicit sentiment, and negation scope ambiguity as the primary remaining challenges.

---