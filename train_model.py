import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib # Used to save the model and vectorizer

print("Starting the model training process...")

# 1. Load Data
df = pd.read_csv('IMDB Dataset.csv')

# 2. Clean Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['cleaned_review'] = df['review'].apply(clean_text)

# 3. Prepare Data
df['sentiment_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X = df['cleaned_review']
y = df['sentiment_label']

# 4. Vectorize Text
# We train the vectorizer on the ENTIRE dataset for a more robust vocabulary
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# 5. Train the Model
model = LogisticRegression(solver='liblinear')
model.fit(X_tfidf, y)

print("Model training complete.")

# 6. Save the Model and Vectorizer
# joblib is efficient for saving objects with large numpy arrays
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

print("Model and vectorizer have been saved as 'sentiment_model.joblib' and 'tfidf_vectorizer.joblib'")