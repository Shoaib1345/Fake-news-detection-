import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

false_data = pd.read_csv('False_data.csv')
true_data = pd.read_csv('True_data.csv')

false_data['label'] = 'FAKE'
true_data['label'] = 'REAL'

df = pd.concat([false_data, true_data], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

if 'text' not in df.columns:
    df['text'] = df['title'].fillna('')

df['text'] = df['text'].fillna('').astype(str)

print("Dataset shape:", df.shape)
print(df.head())

x_train, x_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tf_train = vectorizer.fit_transform(x_train)
tf_test = vectorizer.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tf_train, y_train)

y_pred_train = pac.predict(tf_train)
train_score = accuracy_score(y_train, y_pred_train)
y_pred_test = pac.predict(tf_test)
test_score = accuracy_score(y_test, y_pred_test)

print(f"\nTraining Accuracy: {round(train_score * 100, 2)}%")
print(f"Testing Accuracy: {round(test_score * 100, 2)}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test, labels=["FAKE", "REAL"]))


model_filename = "finalized_model.pkl"
pickle.dump(pac, open(model_filename, "wb"))
print(f"\nModel saved to {model_filename}")

vectorizer_filename = "tfidf_vectorizer.pkl"
pickle.dump(vectorizer, open(vectorizer_filename, "wb"))
print(f"Vectorizer saved to {vectorizer_filename}")

