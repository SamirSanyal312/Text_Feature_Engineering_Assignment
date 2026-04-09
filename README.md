# Text Feature Engineering Assignment Report

## 1. Project Overview

This project builds a text processing pipeline on real-world product reviews collected from Flipkart. The goal was to convert raw user-generated review text into numerical features suitable for machine learning. The project includes data collection, preprocessing, vocabulary creation, feature engineering using One Hot Encoding, Bag of Words, and TF-IDF, comparison analysis, sparse matrix analysis, and a mini sentiment classification use case.

---

## 2. Dataset Collection

Product reviews were scraped from Flipkart using Selenium. The review data was stored in CSV format with the main column:

- `review_text`

Initially, the scraped dataset was highly positive, so additional negative review samples were added to support sentiment classification.

### Final dataset summary
- Total reviews after balancing: **130**
- Positive reviews: **118**
- Negative reviews: **10**
- Neutral reviews: **2**

---

## 3. Preprocessing

The following preprocessing steps were applied to the review text:

1. Converted text to lowercase
2. Tokenized the text
3. Removed punctuation
4. Removed stopwords
5. Applied lemmatization

Additional cleaning steps:
- Extracted rating values from review text
- Removed rating prefixes from the review body
- Created tokenized and cleaned text columns

### Example
**Original review:**  
`5.0 • Mind-blowing purchase. Super and cool photo`

**After preprocessing:**  
- Tokens: `['mindblowing', 'purchase', 'super', 'cool', 'photo']`
- Clean text: `mindblowing purchase super cool photo`

---

## 4. Vocabulary Creation

Vocabulary was created in two ways:

- Manually from the cleaned text
- Using `CountVectorizer` from sklearn

### Vocabulary sizes
- Manual vocabulary size: **391**
- Sklearn vocabulary size: **388**

The small difference is due to sklearn's built-in tokenization and filtering rules.

### Top frequent words
Some of the most frequent words in the dataset were:

- quality
- love
- also
- wow
- amazing
- perfect
- fabulous

These words indicate that the dataset is dominated by positive product opinions.

---

## 5. Feature Engineering

Three feature engineering methods were implemented:

### 5.1 One Hot Encoding
Represents each review using binary values:
- `1` if a word is present
- `0` if a word is absent

**Shape:** `(120, 391)`

### 5.2 Bag of Words
Represents each review using raw word frequency counts.

**Shape:** `(120, 388)`

### 5.3 TF-IDF
Represents each review using weighted word importance scores.

**Shape:** `(120, 388)`

---

## 6. Comparison of OHE, BoW, and TF-IDF

| Technique | Representation | Uses Frequency | Captures Importance | Main Limitation |
|----------|----------------|----------------|---------------------|-----------------|
| One Hot Encoding | Binary presence/absence | No | No | Ignores frequency and importance |
| Bag of Words | Word counts | Yes | Limited | Common words can dominate |
| TF-IDF | Weighted importance | Yes | Yes | Does not capture semantic meaning or context |

### Observation
- One Hot Encoding is simple but limited
- Bag of Words captures counts but not importance
- TF-IDF is more informative because it highlights useful words

---

## 7. TF-IDF Analysis

The most important TF-IDF words in this dataset included:

- good
- product
- great
- phone
- best
- super
- awesome
- terrific
- purchase
- camera

These words are important because they strongly reflect customer opinions and product-related features.

### Why common words usually get lower weight
In general, TF-IDF reduces the weight of words that appear in many documents because they are less useful for distinguishing one document from another.

### Special note for this dataset
In this project, the dataset was limited and highly positive, so some common positive words such as `good` and `great` still received relatively high TF-IDF scores. This is because they remained informative within the dataset even though they appeared frequently.

---

## 8. Sparse Matrix Analysis

All three text representations produced highly sparse matrices.

### Sparsity results

| Matrix | Shape | Sparsity (%) |
|--------|-------|--------------|
| OHE | (120, 391) | 97.866581 |
| BoW | (120, 388) | 97.858677 |
| TF-IDF | (120, 388) | 97.858677 |

### Observation
These matrices are sparse because each review contains only a small subset of the full vocabulary, so most entries are zero.

### Why sparse matrices are inefficient for large-scale systems
Sparse matrices waste memory if stored densely and can become computationally expensive when the number of documents and features becomes very large.

---

## 9. Real-World Discussion

### Why Bag of Words fails to understand semantics
Bag of Words only counts words and ignores context, word order, and meaning. For example:

- `This phone is good`
- `This phone is not good`

These two sentences may look similar in BoW even though their meanings are different.

### When to use Bag of Words and TF-IDF in industry
- **Bag of Words:** useful for simple baseline models, text classification, and fast experiments
- **TF-IDF:** useful for search, information retrieval, document ranking, and classical NLP tasks

### Limitations of TF-IDF
- Does not understand context
- Cannot capture sarcasm or word order
- Does not understand semantic similarity
- Produces sparse feature matrices

---

## 10. Mini Use Case: Sentiment Classification

Sentiment labels were created from ratings:

- `rating >= 4` → positive
- `rating <= 2` → negative
- `rating == 3` → neutral

Neutral reviews were removed for binary classification.

### Class distribution
- Positive: **118**
- Negative: **10**
- Neutral: **2**

### Models tested
- Logistic Regression + BoW
- Logistic Regression + TF-IDF
- Naive Bayes + BoW
- Naive Bayes + TF-IDF

### Results

| Model | Accuracy |
|------|----------|
| Naive Bayes + BoW | 0.961538 |
| Logistic Regression + BoW | 0.923077 |
| Logistic Regression + TF-IDF | 0.923077 |
| Naive Bayes + TF-IDF | 0.923077 |

### Interpretation
The best accuracy was achieved by **Naive Bayes + BoW**. However, the dataset was highly imbalanced, with far more positive reviews than negative ones.

For Logistic Regression, the model failed to correctly predict any negative reviews, which caused:
- negative precision = 0.00
- negative recall = 0.00
- negative f1-score = 0.00

This shows that **accuracy alone can be misleading** in imbalanced classification tasks.

---

## 11. Challenges Faced

Some challenges faced during the project were:

- Dynamic content loading during review scraping
- Changing Flipkart page structure
- Need for debugging HTML and selectors
- Strong class imbalance in scraped sentiment data
- TF-IDF interpretation on a limited, highly positive dataset

---

## 12. Conclusion

This project successfully implemented a complete text feature engineering pipeline on real-world review data. The workflow included scraping, preprocessing, vocabulary creation, One Hot Encoding, Bag of Words, TF-IDF, comparison analysis, sparse matrix analysis, and sentiment classification.

The results show that TF-IDF provides more meaningful weighting than One Hot Encoding and Bag of Words, while all classical methods produce highly sparse matrices. The sentiment classification task also highlighted an important practical issue: class imbalance can significantly affect model performance and interpretation.

Overall, this project demonstrates both the usefulness and the limitations of traditional text feature engineering techniques in real-world NLP tasks.

---

## 13. Files in Repository

- `web_scrapper.py` – review scraping script
- `preprocessing.py` – preprocessing code
- `vocab.py` – vocabulary creation
- `feat_eng.py` – feature engineering
- `analysis.py` – comparison and sparsity analysis
- `senti_clss.py` – sentiment classification
- `flipkart_reviews.csv` – raw scraped data
- `flipkart_reviews_preprocessed.csv` – cleaned dataset
- `one_hot_encoded.csv`
- `bag_of_words.csv`
- `tfidf_features.csv`
- `comparison_table.csv`
- `tfidf_word_importance.csv`
- `sparsity_analysis.csv`
- `sentiment_model_comparison.csv`

---