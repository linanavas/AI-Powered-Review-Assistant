import streamlit as st
import pandas as pd
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
import plotly.express as px
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

#----------------------------------------------------------------------
# Load data
#----------------------------------------------------------------------

input_path = '/content/drive/MyDrive/Final Project BBS/data/'
df_games = pd.read_csv(f"{input_path}games_data_06122025.csv", encoding = "latin1")
game_list = sorted(df_games["Name"].unique())
df_reviews = pd.read_csv(f"{input_path}steam_reviews_summarizer.csv")

#----------------------------------------------------------------------
# Preprocess data
#----------------------------------------------------------------------

df_games["Price"] = np.where(df_games["Price"] == "Free", 0, df_games["Price"])
df_games["Price"] = df_games["Price"].astype(float)
df_games["Release"] = pd.to_datetime(df_games["Release"])

df_reviews['date'] = pd.to_datetime(df_reviews['date'])

st.title("🎮 AI Gaming Reviews Assistant")
st.write("This application uses Natural Language Processing to analyze game reviews.")

game_input = st.selectbox("Select a game", game_list)

#----------------------------------------------------------------------
# Game Overview
#----------------------------------------------------------------------

st.header("📊 Game Overview")

game_row = df_games[df_games["Name"] == game_input].iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("Price", f"${game_row['Price']}")
col2.metric("Rating", f"{game_row['Rating']*100:.1f}%")
col3.metric("Release Date", game_row['Release'].strftime("%Y-%m-%d"))

with st.expander("Compare with other games"):
    
    compare_field = st.radio(
        "Compare by:",
        ["Price", "Rating", "Follows"],
        horizontal=True
    )
    if compare_field == "Price" and pd.isna(game_row[compare_field]):
        st.write(f"Price not available for game {game_input}.")

    else:
        # Build histogram
        fig = px.histogram(
            df_games,
            x=compare_field,
            nbins=20,
            opacity=0.7
        )

        # Add marker for selected game
        fig.add_vline(
            x=game_row[compare_field],
            line_width=3,
            line_dash="dash",
            annotation_text=f"{game_input}",
            annotation_position="top",
            line_color="purple"
        )

        st.plotly_chart(fig, use_container_width=True)

#----------------------------------------------------------------------
# Reviews analysis
#----------------------------------------------------------------------

st.header("😊 Sentiment analysis")
df_reviews_game = df_reviews[df_reviews["game"].astype(str) == str(game_input)]

sentiment_game  = (
    df_reviews_game["sentiment_category"]
    .value_counts(normalize = True)
    .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    .reset_index()
    .rename(columns={"sentiment_category": "Sentiment", "proportion": "Proportion"})
)

st.metric("Total Reviews", len(df_reviews_game))
cols = st.columns(3)

cols[0].metric("❤️ Positive", 
                  f"{sentiment_game[sentiment_game["Sentiment"]=="Positive"]["Proportion"][0]*100:.1f}%")
cols[1].metric("😐 Neutral", 
                  f"{sentiment_game[sentiment_game["Sentiment"]=="Neutral"]["Proportion"][1]*100:.1f}%")
cols[2].metric("😡 Negative", 
                  f"{sentiment_game[sentiment_game["Sentiment"]=="Negative"]["Proportion"][2]*100:.1f}%")

# Comparative analysis
with st.expander("Compare with other games"):

    sentiment_all  = (
        df_reviews["sentiment_category"]
        .value_counts(normalize = True)
        .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
        .reset_index()
        .rename(columns={"sentiment_category": "Sentiment", "proportion": "Proportion"})
    )

    sentiment_game["Category"] = game_input
    sentiment_all["Category"] = "All games"
    sentiment_prop = pd.concat([sentiment_game, sentiment_all])

    fig_sentiments = px.bar(
        sentiment_prop,
        y="Proportion",
        x="Sentiment",
        color="Category",
        barmode="stack",
        color_discrete_map={
            game_input: "#440154",
            "All games": "#22A884"
        },
        title = f"{game_input} vs. All games"
    )
    fig_sentiments.update_layout(
        yaxis_title="Percentage of Reviews",
        xaxis_title="Sentiment",
        showlegend=True,
        yaxis=dict(range=[0, 1]),
        plot_bgcolor="white",
    )

    st.plotly_chart(fig_sentiments, use_container_width=True)

# Top aspect analysis
with st.expander("Aspect deep dive"):
    selected_sentiment = st.radio(
        "Select sentiment:",
        ["Any", "Positive", "Neutral", "Negative"],
        horizontal=True
    )

    df_aspects = df_reviews_game.copy()

    if selected_sentiment != "Any":
        df_aspects = df_aspects[df_aspects["sentiment_category"] == selected_sentiment]

    viridis_5 = [
        "#440154",  # deep purple
        "#22A884",  # green-teal
        "#414487",  # indigo/blue
        "#7AD151",   # yellow-green (soft, not neon)
        "#2A788E",  # teal-blue
    ]
    
    aspects_prop = (
        df_aspects["top_aspect"]
        .value_counts()
        .reset_index()
        .rename(columns={"top_aspect": "Top Aspect", "count": "Count"})
    )
    fig_aspects = px.pie(
        aspects_prop,
        names="Top Aspect",
        values="Count",
        color="Top Aspect",
        color_discrete_sequence=viridis_5,
        title="Top Aspect Breakdown",
    )
    st.plotly_chart(fig_aspects, use_container_width=True)

#----------------------------------------------------------------------
# Review summarizer
#----------------------------------------------------------------------

st.header("🧠 AI Review Summarizer")
st.subheader("Query Filters")

# Create the model
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.2,
    openai_api_key=OPENAI_API_KEY
)

# Get filters input

sentiment_input = st.selectbox("Sentiment", 
                               ["Any", "Positive", "Neutral", "Negative"])
aspect_input = st.selectbox("Top Aspect", 
                             ["Any","Graphics", "Performance", "Story", "Gameplay", "Sound", "Price"])
n_reviews = st.number_input("Number of latest reviews to summarize", min_value=1, value=20)

submit = st.button("Generate Summary")

if submit:
    df_filtered = df_reviews_game.copy()
    df_filtered = df_filtered[df_filtered["word_count"] >= 5]

    # Filter by sentiment
    if sentiment_input != "Any":
        df_filtered = df_filtered[df_filtered["sentiment_category"] == sentiment_input]

    # Filter by aspect
    if aspect_input != "Any":
        df_filtered = df_filtered[df_filtered["top_aspect"] == aspect_input]

    # Filter n latest reviews
    df_filtered = df_filtered.sort_values("date", ascending=False)
    selected_reviews = df_filtered.head(n_reviews)

    if selected_reviews.empty:
        st.error("No matching reviews found.")
        st.stop()

    st.write(f"### Found {len(selected_reviews)} reviews. Summarizing...")

    reviews_text = "\n\n".join(
        f"- {row.clean_review}" for _, row in selected_reviews.iterrows()
    )

    # ------------------------------------------------------
    # Prompt for summarization
    # ------------------------------------------------------
    template = """
    You are an AI assistant specializing in summarizing game reviews.
    Summarize the following user-selected reviews focusing on sentiment, main issues, recurring themes, and overall user experience.

    Reviews:
    {reviews}

    Provide a concise but informative summary.
    """
    prompt = PromptTemplate.from_template(template)

    # Run the model
    with st.spinner("Generating summary..."):
        response = llm([HumanMessage(content=prompt.format(reviews=reviews_text))])

    st.subheader("📌 Summary")
    st.write(response.content)
    


