from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

anime_df = pd.read_csv('anime_recsys.csv')  
tfidf_matrix = joblib.load('animevec_tfidf.joblib')  

class AnimeRequest(BaseModel):
    title: str

def get_recommendations(title, n=10):
    try:
        idx = anime_df[anime_df['english_name'] == title].index[0]
    except IndexError:
        raise HTTPException(status_code=404, detail="Anime not found")
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-n-1:-1][::-1]
    recommended_animes = anime_df.iloc[similar_indices][['title']].to_dict(orient='records')
    return recommended_animes

@app.get("/titles")
async def get_anime_titles(skip: int = Query(0), limit: int = Query(1000)):
    titles = anime_df['english_name'].tolist() 
    return titles[skip: skip + limit]  

@app.post("/recommend")
async def recommend_animes(request: AnimeRequest):
    title = request.title
    if title not in anime_df['english_name'].values:
        raise HTTPException(status_code=404, detail="Anime not found")
    
    recommendations = get_recommendations(title)

    description = anime_df[anime_df['english_name'] == title]['description'].values[0]
    japanese_name = anime_df[anime_df['english_name'] == title]['japanese_name'].values[0]
    genres = anime_df[anime_df['english_name'] == title]['genres'].values[0]
    episodes = anime_df[anime_df['english_name'] == title]['episodes'].values[0]

    return {"title": title, "description": description, "japanese_name": japanese_name,
            "genres": genres, "episodes": episodes, "recommendations": recommendations}

