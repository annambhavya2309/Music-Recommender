import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load dataset
print("Loading dataset...")
df = pd.read_csv("spotify_with_lang.csv")

# Features for similarity
feature_cols = ['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo']
X = df[feature_cols]

# Train nearest neighbors model
print("Training model (Nearest Neighbors)...")
model = NearestNeighbors(n_neighbors=15, metric='euclidean')
model.fit(X)

print("System ready! Use inputs to test.\n")

def recommend(song_name, lang=None):
    if song_name not in df['track_name'].values:
        print(f"❌ Song '{song_name}' not found in dataset.")
        return

    print(f"\n🎵 Recommended songs similar to: {song_name}")

    # Locate the input song row
    song_index = df[df['track_name'] == song_name].index[0]
    song_vector = X.iloc[song_index].values.reshape(1, -1)

    # Find K nearest neighbors
    distances, indices = model.kneighbors(song_vector)

    # Convert to list of indices
    neighbor_indices = indices[0][1:]  # skip itself

    # Apply language filter (fallback logic)
    recommendations = df.iloc[neighbor_indices]

    if lang:
        filtered = recommendations[recommendations['language'] == lang]

        if len(filtered) > 0:
            print(f"🌍 Showing {lang} language results:")
            print("-" * 40)
            for _, row in filtered.head(10).iterrows():
                print(f"{row['track_name']} - {row['artists']}")
        else:
            print(f"⚠️ No songs found in language '{lang}'.")
            print("👇 Showing fallback cross-language recommendations:")
            print("-" * 40)
            for _, row in recommendations.head(10).iterrows():
                print(f"{row['track_name']} - {row['artists']} [{row['language']}]")
    else:
        # No language filter applied
        print("-" * 40)
        for _, row in recommendations.head(10).iterrows():
            print(f"{row['track_name']} - {row['artists']} [{row['language']}]")

# User Input Mode
if __name__ == "__main__":
    print("\n🎶 Music Recommendation System\n")
    song = input("▶ Enter song name: ")
    lang = input("🌐 Enter language code (optional, press Enter to skip): ")

    lang = lang.strip() if lang.strip() != "" else None
    recommend(song, lang)
