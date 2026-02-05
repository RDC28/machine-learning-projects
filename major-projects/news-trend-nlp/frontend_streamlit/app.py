
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from streamlit_lottie import st_lottie
import json

# --- Config & Theme Overrides ---
st.set_page_config(
    page_title="News Trend NLP",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def load_css(file_name="style.css"):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css()
except FileNotFoundError:
    pass

import os

# --- Constants ---
DJANGO_API_URL = os.environ.get("DJANGO_API_URL", "http://localhost:8000/api/latest/")
TRIGGER_RUN_URL = os.environ.get("TRIGGER_RUN_URL", "http://localhost:8000/api/run/")

# --- API Helper Functions ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_trends():
    try:
        resp = requests.get(DJANGO_API_URL, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except:
        return None
    return None

def trigger_analysis():
    try:
        resp = requests.post(TRIGGER_RUN_URL, timeout=30)
        return resp.status_code == 200
    except:
        return False

# --- Lottie Loader ---
@st.cache_data(ttl=3600*24)
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# --- Page: Home ---
def page_home():
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("# GLOBAL NARRATIVES")
        st.markdown("""
        <div style='text-align: left; color: #AAA; font-size: 1.1rem; margin-bottom: 20px; line-height: 1.6;'>
        <strong style='color: #FAFAFA;'>Decode the world's news.</strong><br><br>
        We track thousands of global headlines every minute to identify the stories that matter.
        By clustering similar articles, we cut through the noise and show you the underlying trends.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("EXPLORE TRENDS"):
            st.session_state.page = "Trends"
            st.rerun()

    with col2:
        # Tech/Global Network Lottie Animation
        lottie_url = "https://assets9.lottiefiles.com/packages/lf20_M9p23l.json" # AI/Brain Network
        lottie_json = load_lottie_url(lottie_url)
        
        if lottie_json:
            st_lottie(lottie_json, height=350, key="home_anim")
        else:
            # Fallback CSS visual if Lottie fails
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1E2129 0%, #161920 100%); height: 350px; border-radius: 12px; border: 1px solid #333; display: flex; align-items: center; justify-content: center; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
                <div style="text-align: center;">
                    <span style="font-size: 3rem;">🌐</span>
                    <p style="color: #FF0080; font-weight: bold; margin-top: 10px;">GLOBAL INTELLIGENCE</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # High Level Stats
    data = fetch_trends()
    if data and data.get('trends'):
        trends = data['trends']
        c1, c2, c3 = st.columns(3)
        c1.metric("TOP STORIES IDENITIFIED", len(trends))
        c2.metric("SOURCES SCANNED", sum(t['article_count'] for t in trends))
        c3.metric("LATEST UPDATE", data.get('timestamp', 'N/A')[:16].replace('T', ' '))

# --- Page: Trends Explorer ---
def page_trends():
    col_title, col_act = st.columns([3,1])
    with col_title:
        st.markdown("# TREND EXPLORER")
    with col_act:
        if st.button("REFRESH INTELLIGENCE"):
            with st.spinner("Acquiring latest global signals..."):
                success = trigger_analysis()
                if success:
                    st.cache_data.clear()
                    st.rerun()

    data = fetch_trends()
    
    if not data or not data.get('trends'):
        st.info("No intelligence data available. Initialize analysis.")
        return

    trends = data['trends']
    df = pd.DataFrame(trends)

    # Visualization
    st.markdown("### TOPIC VOLUME DISTRIBUTION")
    
    # Dark Mode Plotly
    fig = px.bar(
        df, x='label', y='article_count',
        text_auto=True,
        color='article_count',
        # Low=Light Pink (#FF66B2), High=Neon Pink (#FF0080)
        color_continuous_scale=['#FF66B2', '#FF0080']
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Outfit",
        font_color="#AAA",
        xaxis_title="",
        yaxis_title="Count",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#333'),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### DETAILED NARRATIVES")
    for i, t in enumerate(trends):
        with st.container():
            # Professional Card with Gradient Border
            st.markdown(f"""
            <div style="background-color: #1E2129; padding: 25px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #333; border-left: 4px solid #FF0080; box-shadow: 0 4px 20px rgba(0,0,0,0.4);">
                <h3 style="margin:0; color: #FFF; font-size: 1.3rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                    <span>{i+1}. {t['label'].upper()}</span>
                    <span style="font-size: 0.8rem; color: #FFF; background: linear-gradient(45deg, #FF0080, #FF66B2); padding: 4px 10px; border-radius: 4px; font-weight: 600; box-shadow: 0 2px 10px rgba(255, 0, 128, 0.3);">{t['article_count']} SOURCES</span>
                </h3>
                <div style="margin-bottom: 15px;">
                    <span style="color: #666; font-size: 0.8em; font-family: monospace; letter-spacing: 1px;">DETECTED KEYWORDS:</span>
                    <span style="color: #FF66B2; font-size: 0.9em; font-family: monospace;"> {', '.join(t['keywords'][:5]).upper()}</span>
                </div>
                <p style="font-size: 1.05em; line-height: 1.6; color: #DDD; padding-top: 10px; border-top: 1px solid #333;">{t['summary']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"VIEW SOURCE ARTICLES FOR '{t['label']}'"):
                try:
                    rep_arts = t['representative_articles']
                    if isinstance(rep_arts, str):
                        rep_arts = json.loads(rep_arts)
                    for art in rep_arts:
                            st.markdown(f"""
                            <div style='margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #333;'>
                            <a href="{art['url']}" target="_blank" style="color: #FF66B2; font-weight: 600; font-size: 1em; text-decoration: none;">{art['text']}</a>
                            <br>
                            <span style='color:#666; font-size:0.8em; text-transform: uppercase;'>SOURCE: {art['source']}</span>
                            </div>
                            """, unsafe_allow_html=True)
                except:
                    st.write("Source data unavailable.")

# --- Page: About ---
def page_about():
    st.markdown("# ABOUT THIS PROJECT")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### WHAT IS THIS?
        This is a real-time news intelligence dashboard. It doesn't just show you deadlines; it **reads** them for you.
        
        By analyzing thousands of articles from around the globe, it groups similar stories together (clustering) to identify the major narratives driving the world's conversation right now.

        ### HOW IT WORKS
        1.  **SCAN**: We continuously fetch live news data from the **GDELT Project**, a massive database of global society.
        2.  **CLUSTER**: Our AI algorithms (using **NMF**) automatically group articles that talk about the same topic.
        3.  **SUMMARIZE**: We generate a clear, concise headline for each group so you can grasp the topic instantly.
        
        ### WHY USE IT?
        *   **Escape the Echo Chamber**: See what the world is actually writing about, not just your personalized feed.
        *   **Spot Trends**: Identify emerging stories before they hit the mainstream.
        *   **Save Time**: Read one summary instead of fifty articles.
        """)
        
    with col2:
        st.markdown("""
        <div style="background-color: #1E2129; padding: 20px; border-radius: 8px; border: 1px solid #333;">
            <h4 style="margin-top: 0;">TECHNICAL SPECS</h4>
            <ul style="color: #AAA; list-style-type: square; padding-left: 20px; line-height: 1.8;">
                <li><strong>Data</strong>: GDELT Project v2</li>
                <li><strong>Model</strong>: TF-IDF + NMF Clustering</li>
                <li><strong>Backend</strong>: Django + FastAPI</li>
                <li><strong>Frontend</strong>: Streamlit</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- Main Navigation ---
def main():
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    # Sidebar Nav
    with st.sidebar:
        st.markdown("## NAVIGATION")
        
        if st.button("DASHBOARD HOME", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()
            
        if st.button("TREND EXPLORER", use_container_width=True):
            st.session_state.page = "Trends"
            st.rerun()
            
        if st.button("ABOUT PROJECT", use_container_width=True):
            st.session_state.page = "About"
            st.rerun()
            
        st.markdown("---")
        st.markdown("<div style='text-align: center; color: #444; font-size: 0.8rem;'>v2.1.0 • DARK MODE</div>", unsafe_allow_html=True)

    # Routing
    if st.session_state.page == "Home":
        page_home()
    elif st.session_state.page == "Trends":
        page_trends()
    elif st.session_state.page == "About":
        page_about()

if __name__ == "__main__":
    main()
