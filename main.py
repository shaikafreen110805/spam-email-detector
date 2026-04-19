import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", sep='\t', names=["label", "message"])

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Remove missing values
df.dropna(inplace=True)

# Split data
X = df['message']
y = df['label']

# Convert text to numbers
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    max_features=5000
)
X_tfidf = vectorizer.fit_transform(X)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression(max_iter=1000)
# Train model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
# Test with custom input
print("\n--- Test Your Own Message ---")

user_input = input("Enter a message: ")

# Convert input
input_data = vectorizer.transform([user_input])

# Predict
prediction = model.predict(input_data)[0]

# Probability
prob = model.predict_proba(input_data)[0]

print(f"\nSpam Probability: {prob[1]:.4f}")
print(f"Ham Probability: {prob[0]:.4f}")

# Custom threshold
if prob[1] > 0.4:
    print("Spam message ❌")
else:
    print("Not Spam ✅")