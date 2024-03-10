import pickle
import numpy as np
import pandas as pd
import streamlit as st

df = pd.read_csv("Nigerian Foods.csv")

df["Spice_Level"] = df["Spice_Level"].fillna("No Spice")

df = df.reset_index(drop=True)
indices = pd.Series(df.index, index=df['Food_Name'])

# Load the saved cosine_sim2 matrix
with open('cosine_sim2.pkl', 'rb') as file:
    cosine_sim2 = pickle.load(file)

def get_recommendations(title, cosine_sim=cosine_sim2):
    
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    # print(sim_scores)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar food
    sim_scores = sim_scores[1:11]
    
    food_indices = [i[0] for i in sim_scores]
    
    # Add sim_score to the DataFrame
    df_recommendations = df.iloc[food_indices].copy()
    df_recommendations['similarity'] = [round(score * 100, 2) for score in [i[1] for i in sim_scores]]
    df_recommendations['similarity'] =  df_recommendations['similarity'].apply(lambda x: str(x)+"%")

    return df_recommendations

#print(get_recommendations('Boli'))

st.title("Nigerian Food Recommendation System")

food_selection = st.selectbox("Select a food:", df['Food_Name'])

if food_selection:
  recommendations = get_recommendations(food_selection)
  st.subheader(f"Top 10 Recommendations for {food_selection}:")
  st.dataframe(recommendations)