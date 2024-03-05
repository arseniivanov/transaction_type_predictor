import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
import numpy as np

# Load your data
file_path = 'Transactions.ods'  # Update to your actual file path
data = pd.read_excel(file_path, engine='odf')

# Set the correct headers and remove the header row from the data
data.columns = data.iloc[0]
data = data[1:]

# Normalizing 'Transaktionsdag'
data['Transaktionsdag'] = pd.to_datetime(data['Transaktionsdag'])
day_of_year = data['Transaktionsdag'].dt.dayofyear
days_in_year = 366 if day_of_year.max() > 365 else 365  # Account for leap years
data['Normalized_Transaktionsdag'] = day_of_year / days_in_year

# Embedding 'Referens' using TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=100)  # You can adjust 'max_features' as needed
data['Referens'] = data['Referens'].astype(str)
referens_embeddings = tfidf.fit_transform(data['Referens']).toarray()

# Normalizing 'Belopp'
scaler = MinMaxScaler(feature_range=(0, 1))
data['Normalized_Belopp'] = scaler.fit_transform(data[['Belopp']])
# One-hot encoding 'Label'
encoder = OneHotEncoder()
labels_encoded = encoder.fit_transform(data[['Label']]).toarray()
label_names = encoder.get_feature_names_out(['Label'])

# Preparing the dataset for training
X = np.hstack((data[['Normalized_Transaktionsdag', 'Normalized_Belopp']], referens_embeddings))
y = labels_encoded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
# Use the appropriate metric for multilabel classification
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

# Convert y_pred back from one-hot encoded format to single labels for each sample for comparison
y_pred_labels = np.argmax(y_pred, axis=1)  # Convert predictions to label index
y_test_labels = np.argmax(y_test, axis=1)  # Convert true values to label index

# Now use accuracy_score with the converted labels
print("Accuracy Score:", accuracy_score(y_test_labels, y_pred_labels))

# Additional relevant metrics for multilabel classification
print("Hamming Loss:", hamming_loss(y_test, y_pred))  # Lower is better

# Save the model, encoder, and scaler for later use (optional)
model.save_model('xgb_classifier.json')
# Joblib can be used to save sklearn preprocessor instances like MinMaxScaler and OneHotEncoder
