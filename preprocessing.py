import pandas as pd
import re
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# 1. Download NLTK resources
# -----------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('punkt_tab')
# -----------------------------
# 2. Load dataset
# -----------------------------
df = pd.read_csv("flipkart_reviews.csv")

print("Original Data:")
print(df.head())
print("Shape:", df.shape)

# -----------------------------
# 3. Initialize tools
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# 4. Extract rating
# -----------------------------
def extract_rating(text):
    """
    Extract numeric rating from strings like:
    '5.0 • Mind-blowing purchase. Super and cool photo'
    Returns integer rating if found, else None
    """
    text = str(text)
    match = re.match(r"^\s*(\d+(?:\.\d+)?)", text)
    if match:
        return int(float(match.group(1)))
    return None

# -----------------------------
# 5. Remove rating prefix
# -----------------------------
def remove_rating_prefix(text):
    """
    Removes leading rating and bullet/symbols from review text.
    Example:
    '5.0 • Mind-blowing purchase. Super and cool photo'
    ->
    'Mind-blowing purchase. Super and cool photo'
    """
    text = str(text)

    # remove rating like 5 or 5.0 at start
    text = re.sub(r"^\s*\d+(?:\.\d+)?\s*", "", text)

    # remove bullet/special separators if present
    text = re.sub(r"^[•\-\.:,;|]+\s*", "", text)

    return text.strip()

# -----------------------------
# 6. Preprocessing function
# -----------------------------
def preprocess_text(text, remove_stopwords=True, do_lemmatization=True):
    # convert to lowercase
    text = str(text).lower()

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # tokenization
    tokens = word_tokenize(text)

    # keep alphabetic words only
    tokens = [word for word in tokens if word.isalpha()]

    # optional stopword removal
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]

    # optional lemmatization
    if do_lemmatization:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

# -----------------------------
# 7. Apply rating extraction
# -----------------------------
df["rating"] = df["review_text"].apply(extract_rating)

# -----------------------------
# 8. Remove rating from review text
# -----------------------------
df["review_no_rating"] = df["review_text"].apply(remove_rating_prefix)

# -----------------------------
# 9. Apply preprocessing
# -----------------------------
df["tokens"] = df["review_no_rating"].apply(
    lambda x: preprocess_text(x, remove_stopwords=True, do_lemmatization=True)
)

df["clean_text"] = df["tokens"].apply(lambda x: " ".join(x))

# -----------------------------
# 10. Show output
# -----------------------------
print("\nProcessed Data:")
print(df[["review_text", "rating", "review_no_rating", "tokens", "clean_text"]].head(10))

print("\nRating value counts:")
print(df["rating"].value_counts(dropna=False))

# -----------------------------
# 11. Save processed CSV
# -----------------------------
df.to_csv("flipkart_reviews_preprocessed.csv", index=False, encoding="utf-8")
print("\nSaved to flipkart_reviews_preprocessed.csv")