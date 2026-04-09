import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# -----------------------------
# 1. Load preprocessed file
# -----------------------------
df = pd.read_csv("flipkart_reviews_preprocessed.csv")

print("Data Preview:")
print(df[["review_text", "clean_text"]].head())
print("Shape:", df.shape)

# -----------------------------
# 2. Prepare token lists for OHE
# -----------------------------
# tokens column is saved as string in CSV, so convert it back to list
df["tokens"] = df["tokens"].apply(
    lambda x: x.strip("[]").replace("'", "").replace(",", "").split()
    if pd.notna(x) else []
)

# -----------------------------
# 3. One Hot Encoding
# -----------------------------
mlb = MultiLabelBinarizer()
ohe_matrix = mlb.fit_transform(df["tokens"])

ohe_df = pd.DataFrame(ohe_matrix, columns=mlb.classes_)

print("\nOne Hot Encoding:")
print("OHE Shape:", ohe_df.shape)
print(ohe_df.head())

# save OHE
ohe_df.to_csv("one_hot_encoded.csv", index=False, encoding="utf-8")

# -----------------------------
# 4. Bag of Words
# -----------------------------
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(df["clean_text"])

bow_df = pd.DataFrame(
    X_bow.toarray(),
    columns=bow_vectorizer.get_feature_names_out()
)

print("\nBag of Words:")
print("BoW Shape:", bow_df.shape)
print(bow_df.head())

# save BoW
bow_df.to_csv("bag_of_words.csv", index=False, encoding="utf-8")

# -----------------------------
# 5. TF-IDF
# -----------------------------
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df["clean_text"])

tfidf_df = pd.DataFrame(
    X_tfidf.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("\nTF-IDF:")
print("TF-IDF Shape:", tfidf_df.shape)
print(tfidf_df.head())

# save TF-IDF
tfidf_df.to_csv("tfidf_features.csv", index=False, encoding="utf-8")

# -----------------------------
# 6. Print summary
# -----------------------------
print("\nSaved files:")
print("1. one_hot_encoded.csv")
print("2. bag_of_words.csv")
print("3. tfidf_features.csv")