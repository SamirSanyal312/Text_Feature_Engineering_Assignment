import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# 1. Load preprocessed file
# -----------------------------
df = pd.read_csv("flipkart_reviews_preprocessed.csv")

print("Preprocessed Data:")
print(df.head())
print("Shape:", df.shape)

# -----------------------------
# 2. Manual vocabulary creation
# -----------------------------
all_words = []

for text in df["clean_text"].dropna():
    words = str(text).split()
    all_words.extend(words)

# create vocabulary
vocabulary_manual = sorted(set(all_words))

print("\nManual Vocabulary:")
print(vocabulary_manual[:50])   # print first 50 words only
print("Manual Vocabulary Size:", len(vocabulary_manual))

# -----------------------------
# 3. Word frequency analysis
# -----------------------------
word_freq = Counter(all_words)

print("\nTop 20 Frequent Words:")
for word, freq in word_freq.most_common(20):
    print(f"{word}: {freq}")

# convert top frequent words to dataframe
top_words_df = pd.DataFrame(word_freq.most_common(20), columns=["word", "frequency"])
print("\nTop Words DataFrame:")
print(top_words_df)

# -----------------------------
# 4. Vocabulary creation using sklearn
# -----------------------------
vectorizer = CountVectorizer()
vectorizer.fit(df["clean_text"].dropna())

vocabulary_sklearn = vectorizer.get_feature_names_out()

print("\nSklearn Vocabulary:")
print(vocabulary_sklearn[:50])   # print first 50 words only
print("Sklearn Vocabulary Size:", len(vocabulary_sklearn))

# -----------------------------
# 5. Compare both vocabulary sizes
# -----------------------------
print("\nComparison:")
print("Manual Vocabulary Size :", len(vocabulary_manual))
print("Sklearn Vocabulary Size:", len(vocabulary_sklearn))

# -----------------------------
# 6. Save vocabulary and frequencies
# -----------------------------
vocab_df = pd.DataFrame(vocabulary_manual, columns=["word"])
vocab_df.to_csv("vocabulary_manual.csv", index=False, encoding="utf-8")

freq_df = pd.DataFrame(word_freq.items(), columns=["word", "frequency"])
freq_df = freq_df.sort_values(by="frequency", ascending=False)
freq_df.to_csv("word_frequencies.csv", index=False, encoding="utf-8")

print("\nSaved files:")
print("1. vocabulary_manual.csv")
print("2. word_frequencies.csv")