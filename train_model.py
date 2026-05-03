import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

print("1. Loading massive dataset...")
df = pd.read_csv('dataset.csv')

print("2. Processing text with TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['symptoms'])

le = LabelEncoder()
y = le.fit_transform(df['disease'])

print("3. Splitting data for training and testing...")
# We added stratify=y back in because we have enough data now!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("4. Balancing rare diseases with SMOTE...")
# We removed the k_neighbors hack. SMOTE is running at full power now.
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("5. Training the Random Forest Classifier (This might take a few seconds)...")
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
rf_classifier.fit(X_train_smote, y_train_smote)

print("6. Saving the upgraded AI brain...")
joblib.dump(rf_classifier, 'rf_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("SUCCESS: Professional ML model trained and saved!")