import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1. Load preprocessed file
# -----------------------------
df = pd.read_csv("flipkart_reviews_preprocessed.csv")

print("Original shape:", df.shape)

# -----------------------------
# 2. Create sentiment labels from rating
# -----------------------------
def get_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating <= 2:
        return "negative"
    else:
        return "neutral"

df["sentiment"] = df["rating"].apply(get_sentiment)

print("\nSentiment distribution before filtering:")
print(df["sentiment"].value_counts())

# Keep only positive and negative
df_model = df[df["sentiment"] != "neutral"].copy()

print("\nShape after removing neutral reviews:", df_model.shape)
print(df_model["sentiment"].value_counts())

# -----------------------------
# 3. Define X and y
# -----------------------------
X = df_model["clean_text"]
y = df_model["sentiment"]

# -----------------------------
# 4. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))

# -----------------------------
# 5. Bag of Words + Logistic Regression
# -----------------------------
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

lr_bow = LogisticRegression(max_iter=1000)
lr_bow.fit(X_train_bow, y_train)
y_pred_bow = lr_bow.predict(X_test_bow)

print("\nBoW + Logistic Regression Accuracy:")
print(accuracy_score(y_test, y_pred_bow))

print("\nBoW + Logistic Regression Report:")
print(classification_report(y_test, y_pred_bow))

# -----------------------------
# 6. TF-IDF + Logistic Regression
# -----------------------------
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

lr_tfidf = LogisticRegression(max_iter=1000)
lr_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = lr_tfidf.predict(X_test_tfidf)

print("\nTF-IDF + Logistic Regression Accuracy:")
print(accuracy_score(y_test, y_pred_tfidf))

print("\nTF-IDF + Logistic Regression Report:")
print(classification_report(y_test, y_pred_tfidf))

# -----------------------------
# 7. Optional: Naive Bayes comparison
# -----------------------------
nb_bow = MultinomialNB()
nb_bow.fit(X_train_bow, y_train)
y_pred_nb_bow = nb_bow.predict(X_test_bow)

nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)
y_pred_nb_tfidf = nb_tfidf.predict(X_test_tfidf)

print("\nBoW + Naive Bayes Accuracy:")
print(accuracy_score(y_test, y_pred_nb_bow))

print("\nTF-IDF + Naive Bayes Accuracy:")
print(accuracy_score(y_test, y_pred_nb_tfidf))

# -----------------------------
# 8. Final comparison table
# -----------------------------
results_df = pd.DataFrame({
    "Model": [
        "Logistic Regression + BoW",
        "Logistic Regression + TF-IDF",
        "Naive Bayes + BoW",
        "Naive Bayes + TF-IDF"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred_bow),
        accuracy_score(y_test, y_pred_tfidf),
        accuracy_score(y_test, y_pred_nb_bow),
        accuracy_score(y_test, y_pred_nb_tfidf)
    ]
})

print("\nFinal Model Comparison:")
print(results_df.sort_values(by="Accuracy", ascending=False))

results_df.to_csv("sentiment_model_comparison.csv", index=False, encoding="utf-8")
print("\nSaved to sentiment_model_comparison.csv")