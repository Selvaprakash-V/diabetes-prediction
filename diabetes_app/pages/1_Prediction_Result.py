import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
try:
    from streamlit_extras.switch_page_button import switch_page
except Exception:
    def switch_page(page_name: str):
        try:
            st.session_state['requested_page'] = page_name
        except Exception:
            pass
        try:
            st.sidebar.warning(
                "Optional package 'streamlit-extras' is not installed. Install it with `pip install streamlit-extras` to enable one-click page switching.\n"
                f"Requested page: {page_name} - open it from the sidebar if available."
            )
        except Exception:
            pass

BASE_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BASE_DIR / 'diabetes_app'
MODEL_PATH = APP_DIR / 'model.pkl'
DATA_PATH = BASE_DIR / 'diabetes.csv'
CSS_PATH = APP_DIR / 'assets' / 'premium_style.css'

st.set_page_config(
    page_title="Risk Analysis",
    layout='wide',
    page_icon='📊',
    initial_sidebar_state="collapsed"
)

# Load premium CSS
if CSS_PATH.exists():
    with open(CSS_PATH) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Risk Analysis</div>
    <div class="subtitle">
        Your personalized diabetes risk assessment based on advanced machine learning analysis.
    </div>
</div>
""", unsafe_allow_html=True)

# get input from session
if 'patient_input' not in st.session_state:
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.warning('⚠️ No patient data found. Please complete the assessment on the Home page first.')
    if st.button('← Return to Home'):
        try:
            switch_page('Home')
        except:
            st.info("Please navigate to Home from the sidebar.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

input_data = st.session_state['patient_input']
df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()

def ensure_model():
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        return model, None

    if df.empty:
        st.error('No dataset available to train the fallback model. Add diabetes.csv to project root.')
        st.stop()

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

model, train_accuracy = ensure_model()

# prepare input dataframe
input_df = pd.DataFrame([input_data])

prediction = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0]

# store for diet page
st.session_state['prediction_result'] = int(prediction)
st.session_state['prediction_proba'] = prob.tolist()

healthy_prob = prob[0]*100
diabetic_prob = prob[1]*100

# Main Results Section
st.markdown('<div class="wide-container">', unsafe_allow_html=True)

# Risk Status Card
col_space1, col_main, col_space2 = st.columns([1, 2, 1])
with col_main:
    st.markdown('<div class="premium-card fade-in" style="text-align: center; padding: 3rem;">', unsafe_allow_html=True)
    
    if prediction == 0:
        st.markdown("""
        <div class="icon-wrapper" style="background: linear-gradient(135deg, #34C759 0%, #00C7BE 100%);">
            ✓
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<h2 style="color: #34C759; font-size: 2rem; margin-bottom: 1rem;">Low Risk Profile</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color: #86868B; font-size: 1.125rem;">Your health metrics indicate a low probability of diabetes. Continue maintaining your healthy lifestyle.</p>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="icon-wrapper" style="background: linear-gradient(135deg, #FF9500 0%, #FF3B30 100%);">
            !
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<h2 style="color: #FF9500; font-size: 2rem; margin-bottom: 1rem;">Elevated Risk Profile</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color: #86868B; font-size: 1.125rem;">Your assessment indicates elevated risk factors. We recommend consulting with a healthcare professional for a comprehensive evaluation.</p>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="badge" style="margin-top: 1.5rem; font-size: 1.25rem;">{max(healthy_prob, diabetic_prob):.1f}% Confidence</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Detailed Analysis
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="premium-card fade-in-delay-1">', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; margin-bottom: 2rem;">Probability Distribution</h3>', unsafe_allow_html=True)
    
    # Create elegant donut chart
    fig = go.Figure(data=[go.Pie(
        labels=['Healthy', 'At Risk'],
        values=[healthy_prob, diabetic_prob],
        hole=0.7,
        marker=dict(
            colors=['#007AFF', '#FF9500'],
            line=dict(color='#FFFFFF', width=3)
        ),
        textfont=dict(size=16, color='#1D1D1F', family='SF Pro Display, -apple-system, sans-serif'),
        hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=14, color='#86868B')
        ),
        height=400,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(
            text=f'<b>{max(healthy_prob, diabetic_prob):.0f}%</b>',
            x=0.5, y=0.5,
            font=dict(size=42, color='#1D1D1F', family='SF Pro Display, -apple-system, sans-serif'),
            showarrow=False
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="premium-card fade-in-delay-2">', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; margin-bottom: 2rem;">Key Metrics</h3>', unsafe_allow_html=True)
    
    metrics_data = [
        ("Healthy Probability", f"{healthy_prob:.1f}%", "#007AFF"),
        ("Risk Probability", f"{diabetic_prob:.1f}%", "#FF9500"),
        ("Model Confidence", f"{max(healthy_prob, diabetic_prob):.1f}%", "#34C759")
    ]
    
    for label, value, color in metrics_data:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.8) 0%, rgba(248,249,250,0.8) 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; 
                    border-left: 4px solid {color};">
            <p style="color: #86868B; font-size: 0.875rem; margin: 0; text-transform: uppercase; letter-spacing: 0.05em;">{label}</p>
            <p style="color: #1D1D1F; font-size: 2rem; font-weight: 700; margin: 0.5rem 0 0 0;">{value}</p>
        </div>
        """, unsafe_allow_html=True)
    
    if train_accuracy is not None:
        st.markdown(f"""
        <div style="background: rgba(0, 122, 255, 0.05); padding: 1rem; border-radius: 12px; text-align: center; margin-top: 1rem;">
            <p style="color: #86868B; font-size: 0.875rem; margin: 0;">Model Accuracy (Test Set)</p>
            <p style="color: #007AFF; font-size: 1.5rem; font-weight: 600; margin: 0.5rem 0 0 0;">{train_accuracy*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Population Comparison
if not df.empty:
    st.markdown('<div class="premium-card fade-in-delay-3">', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; margin-bottom: 2rem;">Your Profile vs Population</h3>', unsafe_allow_html=True)
    
    import plotly.express as px
    
    # Create scatter plot
    fig = px.scatter(
        df, 
        x='Age', 
        y='Glucose',
        color='Outcome',
        color_discrete_map={0: '#007AFF', 1: '#FF9500'},
        opacity=0.4,
        labels={'Outcome': 'Status'},
        height=500
    )
    
    # Add user point
    fig.add_scatter(
        x=[input_df['Age'].iloc[0]],
        y=[input_df['Glucose'].iloc[0]],
        mode='markers',
        marker=dict(
            size=20,
            color='#1D1D1F',
            symbol='star',
            line=dict(color='white', width=2)
        ),
        name='Your Profile',
        hovertemplate='<b>Your Profile</b><br>Age: %{x}<br>Glucose: %{y}<extra></extra>'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,0.3)',
        font=dict(family='SF Pro Display, -apple-system, sans-serif', color='#1D1D1F'),
        xaxis=dict(
            title='Age (years)',
            gridcolor='rgba(0,0,0,0.05)',
            showline=False
        ),
        yaxis=dict(
            title='Glucose Level (mg/dL)',
            gridcolor='rgba(0,0,0,0.05)',
            showline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

# Action Buttons
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    st.markdown('<div class="premium-card fade-in-delay-3">', unsafe_allow_html=True)
    
    col_sub1, col_sub2 = st.columns(2, gap="medium")
    
    with col_sub1:
        if st.button('🍎 View Diet Recommendations', use_container_width=True):
            try:
                switch_page('2_Diet_Recommendations')
            except Exception:
                st.info('Please navigate to Diet Recommendations from the sidebar.')
    
    with col_sub2:
        if st.button('← Return to Home', use_container_width=True):
            try:
                switch_page('Home')
            except Exception:
                st.info('Please navigate to Home from the sidebar.')
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #B0B0B5; font-size: 0.875rem; padding: 2rem 0;">
    <p>This assessment is for informational purposes only and does not replace professional medical advice.</p>
    <p style="margin-top: 0.5rem;">Consult a healthcare provider for proper diagnosis and treatment.</p>
</div>
""", unsafe_allow_html=True)
