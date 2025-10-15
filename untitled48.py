

pip install scikit-learn pandas nltk

import nltk
nltk.download('stopwords')

import pandas as pd

# Our dummy dataset
data = {'review': ['The movie was fantastic!',
                   'I absolutely loved it.',
                   'What a terrible film.',
                   'The acting was superb, but the plot was weak.',
                   'I would not recommend this movie.',
                   'It was a complete masterpiece!',
                   'The worst movie I have ever seen.',
                   'Simply brilliant.'],
        'sentiment': ['positive', 'positive', 'negative', 'negative', 'negative', 'positive', 'negative', 'positive']}

df = pd.DataFrame(data)
print(df)

import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Text Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

# Apply the preprocessing function to our DataFrame
df['cleaned_review'] = df['review'].apply(preprocess_text)
print("\nDataFrame with Cleaned Reviews:")
print(df)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to predict the sentiment of a new review
def predict_sentiment(review_text):
    # Preprocess the new review
    cleaned_review = preprocess_text(review_text)
    # Vectorize the cleaned review using the same vectorizer
    vectorized_review = vectorizer.transform([cleaned_review])
    # Predict the sentiment
    prediction = model.predict(vectorized_review)
    return prediction[0]

# Test with a new review
new_review = "The movie was so boring and slow."
predicted_sentiment = predict_sentiment(new_review)

print("\nPrediction on a new review:")
print(f"Review: '{new_review}'")
print(f"Predicted Sentiment: {predicted_sentiment}")

from google.colab import auth
auth.authenticate_user()

#Vasu Kochhar


