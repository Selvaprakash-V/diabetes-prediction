import streamlit as st
from pathlib import Path
import pandas as pd
try:
    from streamlit_extras.switch_page_button import switch_page
except:
    def switch_page(page_name: str):
        try:
            st.session_state['requested_page'] = page_name
        except:
            pass

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / 'diabetes.csv'
APP_DIR = BASE_DIR / 'diabetes_app'
CSS_PATH = APP_DIR / 'assets' / 'premium_style.css'

st.set_page_config(
    page_title="Personalized Diet Plan",
    layout='wide',
    page_icon='🍎',
    initial_sidebar_state="collapsed"
)

# Load premium CSS
if CSS_PATH.exists():
    with open(CSS_PATH) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
else:
    st.error(f"CSS not found: {CSS_PATH}")

# Dark mode state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🥗</div>
        <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700;">Diet Plan</h2>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.875rem; opacity: 0.8;">Nutrition Guidance</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎨 Appearance")
    theme_label = "🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode"
    if st.button(theme_label, use_container_width=True, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🏠 Quick Actions")
    if st.button("⬅️ Back to Home", use_container_width=True):
        switch_page("Home")
    if st.button("📊 View Analysis", use_container_width=True):
        switch_page("1_Prediction_Result")

# Apply dark mode
if st.session_state.dark_mode:
    st.markdown('<script>document.documentElement.setAttribute("data-theme", "dark");</script>', unsafe_allow_html=True)
else:
    st.markdown('<script>document.documentElement.setAttribute("data-theme", "light");</script>', unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Personalized Diet Plan</div>
    <div class="subtitle">
        Tailored nutritional recommendations based on your risk profile to support optimal health.
    </div>
</div>
""", unsafe_allow_html=True)

if 'patient_input' not in st.session_state or 'prediction_result' not in st.session_state:
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.warning('⚠️ Please complete the risk assessment first to receive personalized recommendations.')
    if st.button('← Start Assessment'):
        try:
            switch_page('Home')
        except:
            st.info("Please navigate to Home from the sidebar.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

input_data = st.session_state['patient_input']
pred = int(st.session_state['prediction_result'])
proba = st.session_state.get('prediction_proba', [0,0])

is_high = pred == 1 or proba[1] > 0.5

# Status Overview
st.markdown('<div class="wide-container">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="premium-card fade-in" style="text-align: center; padding: 2rem;">', unsafe_allow_html=True)
    
    risk_color = "#FF9500" if is_high else "#34C759"
    risk_label = "Elevated Risk" if is_high else "Low Risk"
    
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; gap: 1.5rem;">
        <div>
            <p style="color: #86868B; font-size: 0.875rem; margin: 0; text-transform: uppercase;">Risk Status</p>
            <p style="color: {risk_color}; font-size: 1.75rem; font-weight: 700; margin: 0.5rem 0 0 0;">{risk_label}</p>
        </div>
        <div style="width: 2px; height: 50px; background: rgba(0,0,0,0.1);"></div>
        <div>
            <p style="color: #86868B; font-size: 0.875rem; margin: 0; text-transform: uppercase;">Risk Probability</p>
            <p style="color: #1D1D1F; font-size: 1.75rem; font-weight: 700; margin: 0.5rem 0 0 0;">{proba[1]*100:.1f}%</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Diet Recommendations
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown('<div class="premium-card fade-in-delay-1">', unsafe_allow_html=True)
    st.markdown('<h2 style="margin-bottom: 2rem;">Recommended Foods</h2>', unsafe_allow_html=True)
    
    if is_high:
        foods = [
            ("Leafy Greens", "Spinach, kale, collard greens", "Low in carbs, high in fiber and essential nutrients", "🥬"),
            ("Whole Grains", "Quinoa, barley, brown rice", "Complex carbohydrates in moderate portions", "🌾"),
            ("Lean Proteins", "Chicken breast, fish, legumes", "Helps stabilize blood sugar levels", "🐟"),
            ("Nuts & Seeds", "Almonds, chia seeds, flaxseed", "Healthy fats with low glycemic index", "🥜"),
            ("Berries", "Blueberries, strawberries, raspberries", "Lower in sugar, rich in antioxidants", "🫐"),
            ("Fatty Fish", "Salmon, mackerel, sardines", "Omega-3 fatty acids support heart health", "🐠")
        ]
    else:
        foods = [
            ("Balanced Grains", "Oats, brown rice, whole wheat", "Sustained energy from complex carbs", "🌾"),
            ("Colorful Vegetables", "Bell peppers, carrots, broccoli", "Vitamins, minerals, and fiber", "🥦"),
            ("Lean Proteins", "Poultry, fish, beans, tofu", "Essential amino acids for muscle health", "🍗"),
            ("Fresh Fruits", "Apples, oranges, berries", "Natural sugars with fiber in moderation", "🍎"),
            ("Low-Fat Dairy", "Greek yogurt, milk, cheese", "Calcium and protein for bone health", "🥛"),
            ("Healthy Fats", "Avocado, olive oil, nuts", "Support heart and brain function", "🥑")
        ]

    for title, examples, benefit, icon in foods:
        st.markdown(f"""
        <div class="feature-card" style="margin-bottom: 1.25rem;">
            <div style="display: flex; gap: 1rem; align-items: start;">
                <div style="font-size: 2.5rem; min-width: 60px; text-align: center;">{icon}</div>
                <div style="flex: 1;">
                    <h4 style="color: #1D1D1F; margin: 0 0 0.5rem 0;">{title}</h4>
                    <p style="color: #007AFF; font-size: 0.9375rem; margin: 0 0 0.5rem 0; font-weight: 500;">{examples}</p>
                    <p style="color: #86868B; font-size: 0.9375rem; margin: 0; line-height: 1.5;">{benefit}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Quick Tips
    st.markdown('<div class="premium-card fade-in-delay-2">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-bottom: 1.5rem;">Quick Tips</h3>', unsafe_allow_html=True)
    
    if is_high:
        tips = [
            ("🍽️", "Portion Control", "Use smaller plates and measure servings"),
            ("🚫", "Limit Sugars", "Avoid sodas, candies, and processed sweets"),
            ("⏰", "Regular Meals", "Eat at consistent times daily"),
            ("💧", "Stay Hydrated", "Drink 8-10 glasses of water daily"),
            ("👨‍⚕️", "Consult Expert", "Work with a registered dietitian")
        ]
    else:
        tips = [
            ("🥗", "Balanced Meals", "Include all food groups proportionately"),
            ("🏃", "Stay Active", "30 minutes of exercise most days"),
            ("📊", "Monitor Health", "Regular check-ups and screenings"),
            ("😴", "Quality Sleep", "7-9 hours per night"),
            ("🧘", "Manage Stress", "Practice mindfulness and relaxation")
        ]
    
    for icon, title, desc in tips:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(248,249,250,0.9) 100%);
                    padding: 1.25rem; border-radius: 12px; margin-bottom: 1rem;
                    border: 1px solid rgba(0,0,0,0.04);">
            <div style="font-size: 1.75rem; margin-bottom: 0.5rem;">{icon}</div>
            <p style="color: #1D1D1F; font-weight: 600; font-size: 1rem; margin: 0 0 0.25rem 0;">{title}</p>
            <p style="color: #86868B; font-size: 0.875rem; margin: 0; line-height: 1.5;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Foods to Limit
    st.markdown('<div class="premium-card fade-in-delay-3" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-bottom: 1.5rem;">Foods to Limit</h3>', unsafe_allow_html=True)
    
    avoid_foods = [
        "Sugary beverages & sodas",
        "White bread & pastries",
        "Fried & processed foods",
        "High-sodium snacks",
        "Excessive red meat"
    ]
    
    for food in avoid_foods:
        st.markdown(f"""
        <div style="padding: 0.75rem 0; border-bottom: 1px solid rgba(0,0,0,0.06);">
            <p style="color: #FF3B30; margin: 0; font-size: 0.9375rem;">
                <span style="margin-right: 0.5rem;">⚠️</span>{food}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Weekly Goals Section
col_goal1, col_goal2, col_goal3 = st.columns([1, 2, 1])
with col_goal2:
    st.markdown('<div class="premium-card fade-in-delay-3">', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; margin-bottom: 1.5rem;">Set Your Weekly Goals</h3>', unsafe_allow_html=True)
    
    goal_options = [
        '🚶 10,000 steps daily',
        '💪 30 min exercise daily',
        '🚫 No added sugar',
        '🍽️ Portion control',
        '🥗 5 servings of vegetables',
        '💧 8 glasses of water',
        '😴 7-9 hours sleep'
    ]
    
    goals = st.multiselect(
        'Select goals to track this week',
        goal_options,
        help="Choose realistic goals that fit your lifestyle"
    )
    
    if goals:
        st.markdown('<div style="margin-top: 1.5rem; padding: 1.5rem; background: rgba(52, 199, 89, 0.05); border-radius: 12px; border-left: 4px solid #34C759;">', unsafe_allow_html=True)
        st.markdown(f'<p style="color: #34C759; font-weight: 600; margin-bottom: 0.75rem;">✓ {len(goals)} Goals Selected</p>', unsafe_allow_html=True)
        for goal in goals:
            st.markdown(f'<p style="color: #86868B; margin: 0.25rem 0; font-size: 0.9375rem;">• {goal}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Navigation Buttons
col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
with col_nav2:
    st.markdown('<div class="premium-card fade-in-delay-3">', unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns(3, gap="medium")
    
    with col_btn1:
        if st.button('← Home', use_container_width=True):
            try:
                switch_page('Home')
            except:
                st.info('Navigate to Home from sidebar')
    
    with col_btn2:
        if st.button('📊 View Results', use_container_width=True):
            try:
                switch_page('1_Prediction_Result')
            except:
                st.info('Navigate to Results from sidebar')
    
    with col_btn3:
        if st.button('🔄 New Assessment', use_container_width=True):
            # Clear session state
            for key in ['patient_input', 'prediction_result', 'prediction_proba']:
                if key in st.session_state:
                    del st.session_state[key]
            try:
                switch_page('Home')
            except:
                st.info('Navigate to Home from sidebar')
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #B0B0B5; font-size: 0.875rem; padding: 2rem 0;">
    <p>These recommendations are general guidelines. For personalized dietary advice, consult a registered dietitian or healthcare provider.</p>
    <p style="margin-top: 0.5rem;">© 2025 Diabetes Risk Assessment · Premium Health Dashboard</p>
</div>
""", unsafe_allow_html=True)
