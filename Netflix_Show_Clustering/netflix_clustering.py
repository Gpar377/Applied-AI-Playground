import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data creation
data = {
    'title': ['Show1', 'Show2', 'Show3', 'Show4', 'Show5'],
    'genres': [['Drama', 'Romance'], ['Comedy'], ['Drama', 'Thriller'], ['Comedy', 'Romance'], ['Thriller']],
    'rating': [8.5, 7.2, 8.0, 6.5, 7.8],
    'duration': [50, 30, 60, 25, 55]  # in minutes
}

df = pd.DataFrame(data)

# One-hot encode genres
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['genres'])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

# Combine features
features = pd.concat([genre_df, df[['rating', 'duration']]], axis=1)

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

df['cluster'] = clusters

# Plot clusters
sns.scatterplot(data=df, x='rating', y='duration', hue='cluster', palette='viridis')
plt.title('Netflix Show Clusters')
plt.show()
