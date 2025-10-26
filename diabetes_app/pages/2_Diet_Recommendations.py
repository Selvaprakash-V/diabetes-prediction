import streamlit as st
from pathlib import Path
import pandas as pd
from streamlit_extras.switch_page_button import switch_page

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / 'diabetes.csv'

st.set_page_config(page_title="Diet Recommendations", layout='wide', page_icon='🥗')

st.markdown("""
<style>
.title{font-size:26px;color:#2E86AB;font-weight:700}
.card{background:#fff;padding:16px;border-radius:12px;box-shadow:0 8px 20px rgba(0,0,0,0.06)}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🥗 Personalized Diet Recommendations</div>', unsafe_allow_html=True)

if 'patient_input' not in st.session_state or 'prediction_result' not in st.session_state:
    st.warning('Missing patient data or prediction. Please run the flow starting from Home → Prediction.')
    st.stop()

input_data = st.session_state['patient_input']
pred = int(st.session_state['prediction_result'])
proba = st.session_state.get('prediction_proba', [0,0])

# simple recommendation logic
is_high = pred == 1 or proba[1] > 0.5

st.markdown(f"**Risk:** {'High' if is_high else 'Low'} — **Probability:** {proba[1]*100:.1f}%")

col1, col2 = st.columns([2,1])
with col1:
    st.markdown('### Recommended Foods')
    if is_high:
        foods = [
            ("Leafy Greens", "Spinach, kale, collard greens — low-carb, high-fiber"),
            ("Whole Grains (limited)", "Quinoa, barley — moderate portions"),
            ("Lean Proteins", "Chicken, fish, legumes — stabilize blood sugar"),
            ("Nuts & Seeds", "Almonds, chia — healthy fats, low glycemic"),
            ("Berries", "Lower-sugar fruit, rich in antioxidants")
        ]
    else:
        foods = [
            ("Balanced Grains", "Oats, brown rice — good energy sources"),
            ("Fruits & Veg", "Mix of colorful vegetables and fruits in moderation"),
            ("Lean Proteins", "Poultry, fish, legumes"),
            ("Dairy (low-fat)", "Yogurt, milk — calcium and protein"),
            ("Healthy Fats", "Avocado, olive oil, nuts")
        ]

    for title, desc in foods:
        st.markdown(f"<div class='card'><h4>{title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

with col2:
    st.markdown('### Quick Tips')
    if is_high:
        st.warning('Limit simple sugars, prefer complex carbs, and monitor portion sizes. Consider consulting a nutritionist.')
    else:
        st.success('Maintain balanced meals, exercise regularly, and monitor weight and glucose occasionally.')

st.markdown('---')
st.markdown('### Weekly Goals')
goals = st.multiselect('Choose goals to track', ['10k steps/day','30 min exercise','No added sugar','Portion control','Daily vegetables'])
if st.button('🏠 Back to Home'):
    try:
        switch_page('Home')
    except Exception:
        st.experimental_rerun()

if st.button('🔁 Re-enter Patient Info'):
    try:
        switch_page('Home')
    except Exception:
        st.experimental_rerun()
