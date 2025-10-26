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

BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / 'assets' / 'premium_style.css'

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
else:
    st.error(f"CSS file not found at: {CSS_PATH}")

# Dark mode toggle
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Enhanced Sidebar
with st.sidebar:
    # Sidebar header with icon
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🩺</div>
        <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700;">Health Portal</h2>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.875rem; opacity: 0.8;">Advanced Diabetes Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation section
    st.markdown("### 📊 Navigation")
    st.markdown("""
    <div style="padding: 0.5rem 0;">
        <p style="font-size: 0.875rem; opacity: 0.7; margin: 0.5rem 0;">
            Use the menu above to navigate between pages
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Theme toggle
    st.markdown("### 🎨 Appearance")
    col1, col2 = st.columns([3, 1])
    with col1:
        theme_label = "🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode"
        if st.button(theme_label, use_container_width=True, key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📈 Quick Info")
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%); 
                padding: 1rem; border-radius: 12px; margin: 1rem 0;">
        <div style="margin-bottom: 0.75rem;">
            <div style="font-size: 0.875rem; opacity: 0.8;">ML Algorithm</div>
            <div style="font-weight: 600; font-size: 1rem;">Random Forest</div>
        </div>
        <div style="margin-bottom: 0.75rem;">
            <div style="font-size: 0.875rem; opacity: 0.8;">Accuracy</div>
            <div style="font-weight: 600; font-size: 1rem; color: #10B981;">95.2%</div>
        </div>
        <div>
            <div style="font-size: 0.875rem; opacity: 0.8;">Processing Time</div>
            <div style="font-weight: 600; font-size: 1rem;">< 1 second</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Help section
    with st.expander("ℹ️ Help & Support"):
        st.markdown("""
        **How to use:**
        1. Enter your health metrics
        2. Click 'Analyze Risk Profile'
        3. View your results
        4. Get diet recommendations
        
        **Need help?**  
        Contact: support@healthportal.com
        """)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding-top: 1rem; 
                border-top: 1px solid var(--border); font-size: 0.75rem; opacity: 0.6;">
        <p>v2.0.0 | © 2025 Health Portal</p>
    </div>
    """, unsafe_allow_html=True)

# Apply dark mode
if st.session_state.dark_mode:
    st.markdown("""
    <script>
        document.documentElement.setAttribute('data-theme', 'dark');
    </script>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <script>
        document.documentElement.setAttribute('data-theme', 'light');
    </script>
    """, unsafe_allow_html=True)

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

df_path = BASE_DIR.parent / "diabetes.csv"
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

