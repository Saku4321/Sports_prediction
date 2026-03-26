import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date
from scraper import get_news_for_team, get_teams_from_csv
from llm_claude_morale import get_morale_score
from predict import predict_match

st.set_page_config(page_title="Sports Prediction App", page_icon="⚽", layout="wide")
st.title("Football Match Predictor")

st.caption("AI-powered match predictions for the Premier League.")

DATA_PATH = "data/Premier_League/PremierLeague_Match_Data_Ready_For_ML.csv"

DATA_PATH_CS="data/Premier_League/Not_Merged/E0_25_26_LIVE.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df =load_data()
teams=get_teams_from_csv(DATA_PATH)

teams_cs= get_teams_from_csv(DATA_PATH_CS)

tab1, tab2 = st.tabs(["⚽ Prediction", "📈 Team History"])
with tab1:
    col1,col2 = st.columns(2)

    with col1:

        home_team = st.selectbox("Home Team", teams_cs)

    with col2:
        away_team = st.selectbox("Away Team", teams_cs, index = 1)

    if 'result' not in st.session_state:
        st.session_state.result = None

    if st.button("⚡ Analyze match", use_container_width= True):
        if home_team == away_team:
            st.error("Select two different teams")
        else:
            with st.spinner("Fetching news and analyzing morale..."):
                home_news = get_news_for_team(home_team)
                away_news = get_news_for_team(away_team)

                home_headlines = [f"{n['title']} ({n['published']})" for n in home_news]
                away_headlines = [f"{n['title']} ({n['published']})" for n in away_news]

                home_morale = get_morale_score(home_team, home_headlines)
                away_morale = get_morale_score(away_team, away_headlines)
            with st.spinner("Calculating prediction..."):
                st.session_state.result = predict_match(
                    home_team,away_team,
                    home_morale["morale_score"],
                    away_morale["morale_score"]
                )
    if st.session_state.result:
        result = st.session_state.result
        st.subheader("📊 Result Probability")
        home_pct = result["home_win"]
        draw_pct = result["draw"]
        away_pct = result["away_win"]
        st.markdown(f"""
            <div style="margin: 10px 0;">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span style="color:#e74c3c; font-weight:bold;">{home_team}<br>{home_pct:.2f}%</span>
                    <span style="color:#95a5a6; font-weight:bold; text-align:center;">Draw<br>{draw_pct:.2f}%</span>
                    <span style="color:#e67e22; font-weight:bold; text-align:right;">{away_team}<br>{away_pct:.2f}%</span>
                </div>
                <div style="display:flex; height:24px; border-radius:12px; overflow:hidden;">
                    <div style="width:{home_pct:.2f}%; background:#e74c3c;"></div>
                    <div style="width:{draw_pct:.2f}%; background:#95a5a6;"></div>
                    <div style="width:{away_pct:.2f}%; background:#e67e22;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("🧠 Morale Analysis")
        m1, m2 = st.columns(2)
        with m1:
            st.metric(f"{home_team} (Home)", f"{home_morale['morale_score']}/10")
            st.info(home_morale["reasoning"])
        with m2:
            st.metric(f"{away_team} (Away)", f"{away_morale['morale_score']}/10")
            st.info(away_morale["reasoning"])

        st.subheader("📰 News Headlines")
        n1, n2 = st.columns(2)
        with n1:
            st.markdown(f"**{home_team}**")
            for n in home_news:
                st.markdown(f"-{n['title']}")

        with n2:
            st.markdown(f"**{away_team}**")
            for n in away_news:
                st.markdown(f"-{n['title']}")

with tab2:
    t2_col1, t2_col2 = st.columns(2)
    with t2_col1:
        home_team_history = st.selectbox("Home Team", teams, key="history_home")
    with t2_col2:
        away_team_history = st.selectbox("Away Team", teams, key="history_away", index=1)

    st.subheader("📈 ELO History")
    df['Date']=pd.to_datetime(df['Date'])
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    date_range = st.date_input(
        "Time range",
        value=(min_date,max_date),
        min_value=min_date,
        max_value=max_date
    )
    if len(date_range) == 2:
        start,end = date_range
        mask = (df['Date'] >= pd.Timestamp(start)) & (df['Date'] <= pd.Timestamp(end))
        df_filtered = df[mask]
        home_elo = df_filtered[df_filtered['HomeTeam'] == home_team_history][['Date','Home_ELO']].rename(columns={'Home_ELO':'ELO'})
        away_elo = df_filtered[df_filtered['AwayTeam'] == away_team_history][['Date','Away_ELO']].rename(columns={'Away_ELO':'ELO'})

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=home_elo['Date'], y=home_elo['ELO'], name=home_team_history, line=dict(color='#e74c3c')))
        fig2.add_trace(go.Scatter(x=away_elo['Date'], y=away_elo['ELO'], name=away_team_history, line=dict(color='#3498db')))
        fig2.update_layout(
            xaxis_title = "Date",
            yaxis_title = "ELO",
            height = 400,
            margin=dict(l=20, r=20, t=20,b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)