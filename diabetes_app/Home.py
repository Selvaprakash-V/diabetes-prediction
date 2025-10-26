import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
try:
    from streamlit_lottie import st_lottie
except Exception:
    st_lottie = None
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
                "Optional package 'streamlit-extras' is not installed. Install it with `pip install streamlit-extras` to enable one-click page switching.\n" \
                f"Requested page: {page_name} - open it from the sidebar if available."
            )
        except Exception:
            pass

BASE_DIR = Path(__file__).resolve().parents[1]
CSS_PATH = BASE_DIR / 'diabetes_app' / 'assets' / 'premium_style.css'

st.set_page_config(
    page_title="Diabetes Risk Assessment",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="collapsed"
)

# Load premium CSS
if CSS_PATH.exists():
    with open(CSS_PATH) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        Diabetes Risk Assessment
    </div>
    <div class="subtitle">
        Advanced health screening powered by machine learning. 
        Enter your health metrics for a personalized risk evaluation.
    </div>
</div>
""", unsafe_allow_html=True)

df_path = BASE_DIR / "diabetes.csv"
if df_path.exists():
    df = pd.read_csv(df_path)
else:
    df = pd.DataFrame()

def get_defaults():
    if not df.empty:
        return {
            'Pregnancies': int(df['Pregnancies'].median()),
            'Glucose': int(df['Glucose'].median()),
            'BloodPressure': int(df['BloodPressure'].median()),
            'SkinThickness': int(df['SkinThickness'].median()),
            'Insulin': int(df['Insulin'].median()),
            'BMI': float(df['BMI'].median()),
            'DiabetesPedigreeFunction': float(df['DiabetesPedigreeFunction'].median()),
            'Age': int(df['Age'].median())
        }
    return {'Pregnancies':3,'Glucose':120,'BloodPressure':70,'SkinThickness':20,'Insulin':79,'BMI':24.0,'DiabetesPedigreeFunction':0.47,'Age':33}

defaults = get_defaults()

# Centered form container
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<div class="premium-card fade-in-delay-1">', unsafe_allow_html=True)
    
    with st.form("patient_form"):
        st.markdown('<p class="section-title" style="text-align: center; font-size: 1.5rem; margin-bottom: 2rem;">Health Metrics</p>', unsafe_allow_html=True)
        
        # Two column layout for inputs
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### Physiological Factors")
            pregnancies = st.number_input('🤰 Pregnancies', min_value=0, max_value=20, value=defaults['Pregnancies'], step=1, help="Number of times pregnant")
            glucose = st.slider('🍬 Glucose Level', min_value=0, max_value=300, value=defaults['Glucose'], help="Plasma glucose concentration (mg/dL)")
            bp = st.slider('💓 Blood Pressure', min_value=0, max_value=200, value=defaults['BloodPressure'], help="Diastolic blood pressure (mmHg)")
            skin = st.slider('📏 Skin Thickness', min_value=0, max_value=100, value=defaults['SkinThickness'], help="Triceps skin fold thickness (mm)")
        
        with c2:
            st.markdown("#### Metabolic Indicators")
            insulin = st.slider('💉 Insulin Level', min_value=0, max_value=1000, value=defaults['Insulin'], help="2-Hour serum insulin (μU/mL)")
            bmi = st.number_input('⚖️ BMI', min_value=0.0, max_value=100.0, value=float(defaults['BMI']), step=0.1, format="%.1f", help="Body mass index (kg/m²)")
            dpf = st.number_input('🧬 Pedigree Function', min_value=0.0, max_value=5.0, value=float(defaults['DiabetesPedigreeFunction']), step=0.01, format="%.2f", help="Diabetes pedigree function")
            age = st.number_input('🎂 Age', min_value=0, max_value=120, value=defaults['Age'], help="Age in years")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Progress indicator
        completeness = int(np.mean([
            pregnancies/20*100,
            glucose/300*100,
            bp/200*100,
            (skin/100)*100,
            min(insulin,300)/300*100,
            min(bmi,50)/50*100,
            min(dpf,2.5)/2.5*100,
            age/120*100
        ]))
        
        st.markdown(f'<p style="text-align: center; color: #86868B; font-size: 0.875rem; margin-bottom: 0.5rem;">Form Completion: {completeness}%</p>', unsafe_allow_html=True)
        st.progress(completeness / 100)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Analyze Risk Profile", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    input_data = {
        'Pregnancies': int(pregnancies),
        'Glucose': int(glucose),
        'BloodPressure': int(bp),
        'SkinThickness': int(skin),
        'Insulin': int(insulin),
        'BMI': float(bmi),
        'DiabetesPedigreeFunction': float(dpf),
        'Age': int(age)
    }
    st.session_state['patient_input'] = input_data
    
    # Success message
    st.markdown('<div class="premium-card fade-in" style="margin-top: 2rem; text-align: center;">', unsafe_allow_html=True)
    st.success('✓ Data saved successfully. Proceeding to analysis...')
    st.markdown('</div>', unsafe_allow_html=True)
    
    try:
        switch_page('1_Prediction_Result')
    except Exception:
        st.info("Please navigate to the **Prediction Result** page from the sidebar.")

# Dataset insights section
st.markdown("<br><br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<div class="premium-card fade-in-delay-2">', unsafe_allow_html=True)
    with st.expander("� Dataset Insights", expanded=False):
        if not df.empty:
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Total Cases", f"{len(df):,}")
            with metric_col2:
                st.metric("Diabetic", f"{int(df['Outcome'].sum()):,}")
            with metric_col3:
                st.metric("Healthy", f"{int(len(df) - df['Outcome'].sum()):,}")
            with metric_col4:
                st.metric("Risk Rate", f"{df['Outcome'].mean()*100:.1f}%")
        else:
            st.info("Dataset not available. Place `diabetes.csv` in the project root for enhanced insights.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #B0B0B5; font-size: 0.875rem; padding: 2rem 0;">
    <p>Powered by advanced machine learning · Designed for healthcare professionals</p>
</div>
""", unsafe_allow_html=True)

