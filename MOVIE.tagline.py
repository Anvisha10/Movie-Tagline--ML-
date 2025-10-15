import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG --- 
# Adjust this to the folder where you unzipped the Kaggle dataset
dataset_path = "c:/Users/PC/AppData/Local/Temp/Rar$DRa6668.12082/movies_metadata.csv"

# Load dataset
print("Loading dataset...")
df = pd.read_csv(dataset_path, low_memory=False)

# Keep only rows with non-empty taglines and genres
df = df[df['tagline'].notna() & (df['tagline'].str.strip() != "")]
df = df[df['genres'].notna() & (df['genres'].str.strip() != "")]

# Extract first genre from the genres JSON-like string
def get_first_genre(genres_str):
    try:
        genres_list = ast.literal_eval(genres_str)
        if genres_list:
            return genres_list[0]['name']
    except:
        return None

df['main_genre'] = df['genres'].apply(get_first_genre)
df = df[df['main_genre'].notna()]

print(f"Dataset size after cleaning: {df.shape}")

# Feature engineering: tagline length
df['tagline_length'] = df['tagline'].apply(len)

# --- Bar plot of mean tagline length by genre ---
mean_lengths = df.groupby('main_genre')['tagline_length'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=mean_lengths.index, y=mean_lengths.values, palette='viridis')
plt.xticks(rotation=90)
plt.ylabel('Mean Tagline Length')
plt.title('Mean Tagline Length by Genre')
plt.tight_layout()
plt.show()

# Prepare features and labels
X = df['tagline']
y = df['main_genre']

# Vectorize taglines
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_vec = vectorizer.fit_transform(X)

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=clf.classes_, yticklabels=clf.classes_, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()
