import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from fastapi import FastAPI
import uvicorn

# Load and preprocess data (run once)
df = pd.read_csv('spotify_millsongdata.csv')

def clean_text(text):
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

df['clean_text'] = df['text'].apply(clean_text)
df['combined'] = df['artist'] + ' ' + df['song'] + ' ' + df['clean_text']

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined'])

# Recommendation function
def recommend_songs(song_name: str, n: int = 5):
    idx = df[df['song'].str.lower().str.contains(song_name.lower())].index
    if len(idx) == 0:
        return {"error": "Song not found"}
    idx = idx[0]  # use the first match

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-n-1:-1][::-1]  # exclude itself

    recs = df.iloc[top_indices][['artist', 'song']].to_dict('records')
    return {"recommendations": recs}

# FastAPI app
app = FastAPI(title="Song Recommendation System")

@app.get("/")
def root():
    return {"message": "Song Recommendation API is running. Use /recommend?song_name=YourSong"}

@app.get("/recommend")
def get_recommendations(song_name: str, n: int = 5):
    return recommend_songs(song_name, n)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)