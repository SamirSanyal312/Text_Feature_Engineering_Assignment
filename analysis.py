import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# -----------------------------
# 1. Load preprocessed data
# -----------------------------
df = pd.read_csv("flipkart_reviews_preprocessed.csv")

# convert tokens string back to list
df["tokens"] = df["tokens"].apply(
    lambda x: x.strip("[]").replace("'", "").replace(",", "").split()
    if pd.notna(x) else []
)

# -----------------------------
# 2. Rebuild feature matrices
# -----------------------------
# One Hot Encoding
mlb = MultiLabelBinarizer()
ohe_matrix = mlb.fit_transform(df["tokens"])

# Bag of Words
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(df["clean_text"])

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df["clean_text"])

# -----------------------------
# 3. Comparison Table
# -----------------------------
comparison_df = pd.DataFrame({
    "Technique": ["One Hot Encoding", "Bag of Words", "TF-IDF"],
    "Representation": [
        "Binary values (0 or 1) for word presence",
        "Word frequency counts",
        "Weighted importance of words"
    ],
    "Uses Frequency": ["No", "Yes", "Yes"],
    "Captures Importance": ["No", "Limited", "Yes"],
    "Sparse Matrix": ["Yes", "Yes", "Yes"],
    "Best For": [
        "Simple presence/absence representation",
        "Basic text classification and word counting",
        "Text mining, search, and feature importance"
    ],
    "Main Limitation": [
        "Ignores frequency and importance",
        "Common words can dominate",
        "Still does not capture semantics or context"
    ]
})

print("\nComparison Table:")
print(comparison_df)

comparison_df.to_csv("comparison_table.csv", index=False, encoding="utf-8")

# -----------------------------
# 4. TF-IDF Important Words
# -----------------------------
mean_tfidf_scores = np.asarray(X_tfidf.mean(axis=0)).ravel()
tfidf_words = tfidf_vectorizer.get_feature_names_out()

tfidf_importance_df = pd.DataFrame({
    "word": tfidf_words,
    "mean_tfidf_score": mean_tfidf_scores
}).sort_values(by="mean_tfidf_score", ascending=False)

print("\nTop 20 Important Words in TF-IDF:")
print(tfidf_importance_df.head(20))

tfidf_importance_df.to_csv("tfidf_word_importance.csv", index=False, encoding="utf-8")

# -----------------------------
# 5. Sparse Matrix Analysis
# -----------------------------
def calculate_sparsity(matrix):
    total_elements = matrix.shape[0] * matrix.shape[1]
    
    if hasattr(matrix, "count_nonzero"):
        non_zero = matrix.count_nonzero()
    else:
        non_zero = np.count_nonzero(matrix)

    zero_elements = total_elements - non_zero
    sparsity = (zero_elements / total_elements) * 100

    return total_elements, non_zero, zero_elements, sparsity

# OHE sparsity
ohe_total = ohe_matrix.shape[0] * ohe_matrix.shape[1]
ohe_non_zero = np.count_nonzero(ohe_matrix)
ohe_zero = ohe_total - ohe_non_zero
ohe_sparsity = (ohe_zero / ohe_total) * 100

# BoW sparsity
bow_total, bow_non_zero, bow_zero, bow_sparsity = calculate_sparsity(X_bow)

# TF-IDF sparsity
tfidf_total, tfidf_non_zero, tfidf_zero, tfidf_sparsity = calculate_sparsity(X_tfidf)

sparsity_df = pd.DataFrame({
    "Matrix": ["OHE", "BoW", "TF-IDF"],
    "Shape": [str(ohe_matrix.shape), str(X_bow.shape), str(X_tfidf.shape)],
    "Total Elements": [ohe_total, bow_total, tfidf_total],
    "Non-Zero Elements": [ohe_non_zero, bow_non_zero, tfidf_non_zero],
    "Zero Elements": [ohe_zero, bow_zero, tfidf_zero],
    "Sparsity (%)": [ohe_sparsity, bow_sparsity, tfidf_sparsity]
})

print("\nSparse Matrix Analysis:")
print(sparsity_df)

sparsity_df.to_csv("sparsity_analysis.csv", index=False, encoding="utf-8")

# -----------------------------
# 6. Final saved files
# -----------------------------
print("\nSaved files:")
print("1. comparison_table.csv")
print("2. tfidf_word_importance.csv")
print("3. sparsity_analysis.csv")