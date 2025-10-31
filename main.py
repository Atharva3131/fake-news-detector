import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
import string
import joblib

# Download stopwords (only the first time)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load datasets
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Label them
true_df["label"] = 0   # 0 = Real
fake_df["label"] = 1   # 1 = Fake

# Combine
data = pd.concat([true_df, fake_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Combine title and text
data["content"] = data["title"].astype(str) + " " + data["text"].astype(str)
data = data[["content", "label"]]
data.dropna(inplace=True)

# Clean text
def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data["content"] = data["content"].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data["content"], data["label"], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
print("\n📊 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n💾 Model and vectorizer saved successfully!")

# Try a test prediction
sample = "Government announces new technology policy for startups."
sample_vec = vectorizer.transform([sample])
pred = model.predict(sample_vec)[0]
print("\n🔍 Sample Prediction:", "FAKE" if pred == 1 else "REAL")
