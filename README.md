# Movie_Genre_Prediction.
Overview :
This project is a Movie Genre Prediction System that classifies movies into different genres based on their descriptions using Natural Language Processing (NLP) and Machine Learning. It employs TF-IDF vectorization and a Naïve Bayes classifier to perform text classification.

Features:
Loads and preprocesses movie description datasets
Extracts features using TF-IDF Vectorization
Trains a Multinomial Naïve Bayes model for classification
Evaluates model accuracy using precision, recall, and F1-score
Predicts genres for unseen movie descriptions

Dataset:
The dataset consists of movie titles, descriptions, and genres:

ID ::: TITLE ::: GENRE ::: DESCRIPTION
1 ::: Inception ::: Sci-Fi ::: A thief who enters the dreams of others to steal secrets.
2 ::: Titanic ::: Romance ::: A young couple falls in love on the ill-fated voyage.

train_data.txt - Contains labeled training data (includes genres)
test_data.txt - Contains test movie descriptions (without genres)
test_data_solution.txt - The correct genres for the test data

Download the dataset from https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

Installation & Setup:
1. Clone the Repository

git clone https://github.com/your-username/Movie-Genre-Prediction.git
cd Movie-Genre-Prediction

2. Install Dependencies
Make sure you have Python installed, then install required libraries:

pip install pandas scikit-learn nltk

3. Download NLTK Resources
   
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

Running the Project:
Place the dataset files in the project folder.
Run the Python script:

python movieGenere.py

Model Training & Evaluation:
Preprocessing: Text is cleaned, tokenized, and lemmatized.
Feature Extraction: TF-IDF vectorization is applied.
Training: The Naïve Bayes classifier is trained on processed data.
Evaluation: Model performance is measured on a validation set.
Prediction: The trained model predicts genres for test data.

Sample Output:

--- Validation Set Evaluation ---
Accuracy Score: 85%
Classification Report:
Precision: 0.87, Recall: 0.84, F1-score: 0.85



