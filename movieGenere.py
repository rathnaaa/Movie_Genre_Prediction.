import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('wordnet')

# 1. Load the Datasets
def load_data(filepath, has_genre=True):
    """
    Loads data from a text file.

    Args:
        filepath (str): The path to the text file.
        has_genre (bool): True if the file contains genre labels (e.g., training data),
                          False if it doesn't (e.g., test data).

    Returns:
        pandas.DataFrame: A DataFrame with columns 'ID', 'TITLE', 'DESCRIPTION', and optionally 'GENRE'.
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(' ::: ')
                if has_genre:
                    if len(parts) == 4:
                        data.append({'ID': parts[0], 'TITLE': parts[1], 'GENRE': parts[2], 'DESCRIPTION': parts[3]})
                    else:
                        print(f"Warning: Malformed line (expected 4 parts): {line.strip()}")
                else:
                    if len(parts) == 3:
                        data.append({'ID': parts[0], 'TITLE': parts[1], 'DESCRIPTION': parts[2]})
                    else:
                        print(f"Warning: Malformed line (expected 3 parts): {line.strip()}")
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

# 2. Preprocess the Text
def preprocess_text(text):
    """
    Cleans and preprocesses the text data.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I)
    text = text.lower()
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

# 3. Feature Extraction
def extract_features(data, max_features=5000):
    """
    Extracts features from the text data using TF-IDF.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(data)
    return features, vectorizer

def extract_test_features(data, vectorizer):
    """
    Extracts features from the test data using a fitted TF-IDF vectorizer.
    """
    features = vectorizer.transform(data)
    return features

# 4. Train a Model
def train_model(features, labels):
    """
    Trains a Multinomial Naive Bayes classifier.
    """
    model = MultinomialNB()
    model.fit(features, labels)
    return model

# 5. Evaluate the Model
def evaluate_model(model, features, labels):
    """
    Evaluates the model's performance.
    """
    predictions = model.predict(features)
    print("Classification Report:\n", classification_report(labels, predictions))
    print("Accuracy Score:", accuracy_score(labels, predictions))

# 6. Predict on Test Data
def predict_genres(model, vectorizer, test_data):
    """
    Predicts genres for the test data.
    """
    test_features = extract_test_features(test_data['DESCRIPTION'], vectorizer)
    predictions = model.predict(test_features)
    return predictions

# --- Main Execution ---

# 1. Load the datasets
train_filepath = "C:/Users/rathn/OneDrive/ドキュメント/GrowthLink/dataset/train_data.txt"
test_filepath = "C:/Users/rathn/OneDrive/ドキュメント/GrowthLink/dataset/test_data.txt"
solution_filepath = "C:/Users/rathn/OneDrive/ドキュメント/GrowthLink/dataset/test_data_solution.txt" # Load solution for comparison

train_data = load_data(train_filepath)
test_data = load_data(test_filepath, has_genre=False)
solution_data = load_data(solution_filepath)

if train_data is None or test_data is None:
    exit()

# Handle potential data loading issues
train_data = train_data.dropna(subset=['DESCRIPTION', 'GENRE'])
test_data = test_data.dropna(subset=['DESCRIPTION'])

# 2. Preprocess the text
train_data['processed_description'] = train_data['DESCRIPTION'].apply(preprocess_text)
test_data['processed_description'] = test_data['DESCRIPTION'].apply(preprocess_text)

# 3. Split training data for model training
X_train, X_val, y_train, y_val = train_test_split(
    train_data['processed_description'], train_data['GENRE'], test_size=0.2, random_state=42
)

# 4. Extract features
X_train_features, vectorizer = extract_features(X_train) # Fit vectorizer on training data
X_val_features = vectorizer.transform(X_val)
X_test_features = vectorizer.transform(test_data['processed_description'])

# 5. Train the model
model = train_model(X_train_features, y_train)

# 6. Evaluate the model on the validation set
print("--- Validation Set Evaluation ---")
evaluate_model(model, X_val_features, y_val)

# 7. Make predictions on the test set
test_predictions = predict_genres(model, vectorizer, test_data)

# Add predictions to the test data DataFrame
test_data['PREDICTED_GENRE'] = test_predictions

# 8. Evaluate against the provided solution
print("\n--- Test Set Evaluation (against solution) ---")
evaluate_model(model, vectorizer.transform(solution_data['DESCRIPTION'].apply(preprocess_text)), solution_data['GENRE'])

# Print or save the test data with predictions
print("\n--- Test Data with Predictions ---")
print(test_data[['ID', 'TITLE', 'DESCRIPTION', 'PREDICTED_GENRE']].head())

# You can save the test data with predictions to a file if needed
# test_data[['ID', 'TITLE', 'DESCRIPTION', 'PREDICTED_GENRE']].to_csv("test_predictions.csv", index=False)