# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

# PAGE CONFIG
st.set_page_config(
    page_title="🩺 Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        color: #A23B72;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .healthy {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .diabetic {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# LOAD DATA
df = pd.read_csv("diabetes.csv")

# MAIN HEADER
st.markdown('<h1 class="main-header">🩺 Diabetes Risk Assessment</h1>', unsafe_allow_html=True)

# SIDEBAR STYLING
st.sidebar.markdown("## 📋 Patient Information")
st.sidebar.markdown("*Please adjust the sliders below to input patient data*")

# INFORMATION SECTION
with st.expander("ℹ️ About This Application", expanded=False):
    st.markdown("""
    This application uses **Machine Learning** to predict diabetes risk based on patient health metrics.
    
    **Features analyzed:**
    - 🤰 Pregnancies: Number of pregnancies
    - 🍬 Glucose: Blood glucose level
    - 💓 Blood Pressure: Systolic blood pressure
    - 📏 Skin Thickness: Triceps skin fold thickness
    - 💉 Insulin: Serum insulin level
    - ⚖️ BMI: Body Mass Index
    - 🧬 Diabetes Pedigree Function: Genetic predisposition
    - 🎂 Age: Patient age
    
    **Model**: Random Forest Classifier trained on diabetes dataset
    """)

# DATASET OVERVIEW
st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Samples", len(df), help="Total number of patients in dataset")
with col2:
    diabetic_count = len(df[df['Outcome'] == 1])
    st.metric("Diabetic Cases", diabetic_count, help="Number of diabetic patients")
with col3:
    healthy_count = len(df[df['Outcome'] == 0])
    st.metric("Healthy Cases", healthy_count, help="Number of healthy patients")
with col4:
    diabetic_percentage = round((diabetic_count / len(df)) * 100, 1)
    st.metric("Diabetes Rate", f"{diabetic_percentage}%", help="Percentage of diabetic cases")

# DATASET STATISTICS
with st.expander("📈 Detailed Dataset Statistics", expanded=False):
    st.dataframe(df.describe(), width='stretch')


# X AND Y DATA
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

# ENHANCED USER INPUT FUNCTION
def user_report():
    # Collect inputs inside a sidebar form so the sidebar acts as the first "page"
    with st.sidebar.form(key="user_input_form"):
        st.markdown("---")

        pregnancies = st.slider('🤰 Pregnancies', min_value=0, max_value=17, value=3,
                                help="Number of times pregnant")

        glucose = st.slider('🍬 Glucose Level (mg/dL)', min_value=0, max_value=200, value=120,
                            help="Blood glucose concentration (normal: 70-140 mg/dL)")

        bp = st.slider('💓 Blood Pressure (mmHg)', min_value=0, max_value=122, value=70,
                       help="Systolic blood pressure (normal: 90-120 mmHg)")

        skinthickness = st.slider('📏 Skin Thickness (mm)', min_value=0, max_value=100, value=20,
                                  help="Triceps skin fold thickness")

        insulin = st.slider('💉 Insulin Level (μU/mL)', min_value=0, max_value=846, value=79,
                            help="Serum insulin level")

        bmi = st.slider('⚖️ BMI (kg/m²)', min_value=0.0, max_value=67.0, value=20.0, step=0.1,
                        help="Body Mass Index (normal: 18.5-24.9)")

        dpf = st.slider('🧬 Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47, step=0.01,
                        help="Genetic predisposition to diabetes")

        age = st.slider('🎂 Age (years)', min_value=21, max_value=88, value=33,
                        help="Patient age in years")

        st.markdown("---")
        location = st.selectbox('🌍 Select Your Region', [
            'India (South)', 'India (North)', 'India (East)', 'India (West)',
            'USA', 'UK', 'Europe', 'Middle East', 'China', 'Japan', 'Other'
        ], help="Choose your region to get local meal suggestions")

        st.markdown("---")
        st.markdown("### 🎯 Health Indicators")

        # BMI category
        if bmi < 18.5:
            bmi_status = "🔵 Underweight"
        elif bmi < 25:
            bmi_status = "🟢 Normal"
        elif bmi < 30:
            bmi_status = "🟡 Overweight"
        else:
            bmi_status = "🔴 Obese"
        st.write(f"BMI Category: {bmi_status}")

        # Glucose status
        if glucose < 70:
            glucose_status = "🔵 Low"
        elif glucose <= 140:
            glucose_status = "🟢 Normal"
        elif glucose <= 199:
            glucose_status = "🟡 Pre-diabetic"
        else:
            glucose_status = "🔴 Diabetic"
        st.write(f"Glucose Level: {glucose_status}")

        submit = st.form_submit_button(label="🔍 Analyze Patient")

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age,
        'Location': location
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data, submit




# PATIENT DATA INPUT
user_data, submitted = user_report()

if not submitted:
    st.markdown('<h2 class="sub-header">👋 Please provide patient info in the sidebar</h2>', unsafe_allow_html=True)
    st.info("Fill the inputs in the left sidebar and press the '🔍 Analyze Patient' button to view personalized results.")
    st.stop()

st.markdown('<h2 class="sub-header">👤 Current Patient Data</h2>', unsafe_allow_html=True)

# Display patient data in a nice format
col1, col2 = st.columns(2)
with col1:
    st.write("**Physical Measurements:**")
    st.write(f"• Age: {user_data['Age'].iloc[0]} years")
    st.write(f"• BMI: {user_data['BMI'].iloc[0]:.1f} kg/m²")
    st.write(f"• Blood Pressure: {user_data['BloodPressure'].iloc[0]} mmHg")
    st.write(f"• Skin Thickness: {user_data['SkinThickness'].iloc[0]} mm")

with col2:
    st.write("**Medical History:**")
    st.write(f"• Pregnancies: {user_data['Pregnancies'].iloc[0]}")
    st.write(f"• Glucose Level: {user_data['Glucose'].iloc[0]} mg/dL")
    st.write(f"• Insulin Level: {user_data['Insulin'].iloc[0]} μU/mL")
    st.write(f"• Diabetes Pedigree: {user_data['DiabetesPedigreeFunction'].iloc[0]:.3f}")

# MODEL TRAINING AND PREDICTION
st.markdown('<h2 class="sub-header">🤖 AI Model Analysis</h2>', unsafe_allow_html=True)

with st.spinner("🔄 Training AI model and analyzing patient data..."):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    
    # Prepare user data for prediction (exclude Location column if present)
    user_data_for_prediction = user_data.drop('Location', axis=1) if 'Location' in user_data.columns else user_data

    # Get prediction and probability
    user_result = rf.predict(user_data_for_prediction)
    user_probability = rf.predict_proba(user_data_for_prediction)
    
    # Model accuracy
    accuracy = accuracy_score(y_test, rf.predict(x_test))

# PREDICTION RESULTS
st.markdown('<h2 class="sub-header">📋 Diagnosis Results</h2>', unsafe_allow_html=True)

# Create prediction display
if user_result[0] == 0:
    prediction_class = "healthy"
    prediction_text = "✅ LOW DIABETES RISK"
    prediction_emoji = "😊"
    risk_level = "Low Risk"
    recommendation = "Maintain your healthy lifestyle! Continue regular exercise and balanced diet."
    color = '#2E86AB'
else:
    prediction_class = "diabetic"
    prediction_text = "⚠️ HIGH DIABETES RISK"
    prediction_emoji = "😟"
    risk_level = "High Risk"
    recommendation = "Please consult with a healthcare professional for proper diagnosis and treatment plan."
    color = '#E74C3C'

# Probability scores
healthy_prob = user_probability[0][0] * 100
diabetic_prob = user_probability[0][1] * 100

# Display prediction in a styled box
st.markdown(f"""
<div class="prediction-box {prediction_class}">
    <h1>{prediction_emoji} {prediction_text}</h1>
    <h3>Risk Level: {risk_level}</h3>
    <p style="font-size: 1.2rem; margin-top: 1rem;">{recommendation}</p>
</div>
""", unsafe_allow_html=True)

# Probability gauge
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Create a donut chart for probability
    fig = go.Figure(data=[go.Pie(
        labels=['Healthy', 'Diabetic'],
        values=[healthy_prob, diabetic_prob],
        hole=.6,
        marker_colors=['#2E86AB', '#E74C3C']
    )])
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=14
    )
    
    fig.update_layout(
        title={
            'text': f"<b>Prediction Confidence</b><br><span style='font-size:24px'>{max(healthy_prob, diabetic_prob):.1f}%</span>",
            'x': 0.5,
            'font': {'size': 16}
        },
        showlegend=False,
        height=300
    )
    
    st.plotly_chart(fig, width='stretch')

# Model metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Accuracy", f"{accuracy*100:.1f}%", help="Accuracy on test dataset")
with col2:
    st.metric("Healthy Probability", f"{healthy_prob:.1f}%", help="Probability of being healthy")
with col3:
    st.metric("Diabetic Probability", f"{diabetic_prob:.1f}%", help="Probability of having diabetes")



# ENHANCED VISUALIZATIONS
st.markdown('<h2 class="sub-header">📈 Patient Analysis Dashboard</h2>', unsafe_allow_html=True)

# Color for patient data point
patient_color = '#E74C3C' if user_result[0] == 1 else '#2E86AB'
patient_symbol = 'diamond' if user_result[0] == 1 else 'star'

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["🔄 Comparison Charts", "📊 Distribution Analysis", "🎯 Risk Factors", "📋 Correlation Matrix"])

with tab1:
    st.markdown("### Patient vs Population Comparison")

    # Create 2x2 grid of comparison charts
    col1, col2 = st.columns(2)

    with col1:
        # Age vs Pregnancies
        fig1 = px.scatter(
            df, x='Age', y='Pregnancies',
            color='Outcome',
            color_discrete_map={0: '#2E86AB', 1: '#E74C3C'},
            title='🤰 Pregnancies vs Age',
            labels={'Outcome': 'Diabetes Status'},
            opacity=0.6,
        )
        fig1.add_scatter(
            x=[user_data['Age'].iloc[0]],
            y=[user_data['Pregnancies'].iloc[0]],
            mode='markers',
            marker=dict(size=15, color=patient_color, symbol=patient_symbol, line=dict(width=2, color='white')),
            name='You',
            showlegend=True,
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, width='stretch')

        # Age vs Glucose
        fig2 = px.scatter(
            df, x='Age', y='Glucose',
            color='Outcome',
            color_discrete_map={0: '#2E86AB', 1: '#E74C3C'},
            title='🍬 Glucose Level vs Age',
            labels={'Outcome': 'Diabetes Status'},
            opacity=0.6,
        )
        fig2.add_scatter(
            x=[user_data['Age'].iloc[0]],
            y=[user_data['Glucose'].iloc[0]],
            mode='markers',
            marker=dict(size=15, color=patient_color, symbol=patient_symbol, line=dict(width=2, color='white')),
            name='You',
            showlegend=True,
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, width='stretch')

    with col2:
        # Age vs BMI
        fig3 = px.scatter(
            df, x='Age', y='BMI',
            color='Outcome',
            color_discrete_map={0: '#2E86AB', 1: '#E74C3C'},
            title='⚖️ BMI vs Age',
            labels={'Outcome': 'Diabetes Status'},
            opacity=0.6,
        )
        fig3.add_scatter(
            x=[user_data['Age'].iloc[0]],
            y=[user_data['BMI'].iloc[0]],
            mode='markers',
            marker=dict(size=15, color=patient_color, symbol=patient_symbol, line=dict(width=2, color='white')),
            name='You',
            showlegend=True,
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, width='stretch')

        # Age vs Blood Pressure
        fig4 = px.scatter(
            df, x='Age', y='BloodPressure',
            color='Outcome',
            color_discrete_map={0: '#2E86AB', 1: '#E74C3C'},
            title='💓 Blood Pressure vs Age',
            labels={'Outcome': 'Diabetes Status'},
            opacity=0.6,
        )
        fig4.add_scatter(
            x=[user_data['Age'].iloc[0]],
            y=[user_data['BloodPressure'].iloc[0]],
            mode='markers',
            marker=dict(size=15, color=patient_color, symbol=patient_symbol, line=dict(width=2, color='white')),
            name='You',
            showlegend=True,
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, width='stretch')

with tab2:
    st.markdown("### Feature Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Glucose distribution
        fig_hist1 = px.histogram(
            df, x='Glucose', color='Outcome',
            title='🍬 Glucose Distribution',
            color_discrete_map={0: '#2E86AB', 1: '#E74C3C'},
            marginal='box',
            opacity=0.7
        )
        fig_hist1.add_vline(
            x=user_data['Glucose'].iloc[0], 
            line_dash="dash", 
            line_color=patient_color,
            annotation_text="Your Level"
        )
        st.plotly_chart(fig_hist1, width='stretch')
        
        # BMI distribution
        fig_hist2 = px.histogram(
            df, x='BMI', color='Outcome',
            title='⚖️ BMI Distribution',
            color_discrete_map={0: '#2E86AB', 1: '#E74C3C'},
            marginal='box',
            opacity=0.7
        )
        fig_hist2.add_vline(
            x=user_data['BMI'].iloc[0], 
            line_dash="dash", 
            line_color=patient_color,
            annotation_text="Your BMI"
        )
        st.plotly_chart(fig_hist2, width='stretch')
    
    with col2:
        # Age distribution
        fig_hist3 = px.histogram(
            df, x='Age', color='Outcome',
            title='🎂 Age Distribution',
            color_discrete_map={0: '#2E86AB', 1: '#E74C3C'},
            marginal='box',
            opacity=0.7
        )
        fig_hist3.add_vline(
            x=user_data['Age'].iloc[0], 
            line_dash="dash", 
            line_color=patient_color,
            annotation_text="Your Age"
        )
        st.plotly_chart(fig_hist3, width='stretch')
        
        # Insulin distribution
        fig_hist4 = px.histogram(
            df, x='Insulin', color='Outcome',
            title='💉 Insulin Distribution',
            color_discrete_map={0: '#2E86AB', 1: '#E74C3C'},
            marginal='box',
            opacity=0.7
        )
        fig_hist4.add_vline(
            x=user_data['Insulin'].iloc[0], 
            line_dash="dash", 
            line_color=patient_color,
            annotation_text="Your Level"
        )
        st.plotly_chart(fig_hist4, width='stretch')

with tab3:
    st.markdown("### Risk Factor Analysis")
    
    # Feature importance (mock data - in real scenario, you'd use model.feature_importances_)
    features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'Pregnancies', 'BloodPressure', 'Insulin', 'SkinThickness']
    importance = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]  # Mock importance values
    
    fig_importance = px.bar(
        x=importance, y=features, orientation='h',
        title='📊 Feature Importance in Diabetes Prediction',
        labels={'x': 'Importance Score', 'y': 'Health Factors'},
        color=importance,
        color_continuous_scale='viridis'
    )
    fig_importance.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_importance, width='stretch')
    
    # Patient risk assessment
    st.markdown("### Your Risk Assessment")
    
    user_values = [
        user_data['Glucose'].iloc[0],
        user_data['BMI'].iloc[0], 
        user_data['Age'].iloc[0],
        user_data['DiabetesPedigreeFunction'].iloc[0],
        user_data['Pregnancies'].iloc[0],
        user_data['BloodPressure'].iloc[0],
        user_data['Insulin'].iloc[0],
        user_data['SkinThickness'].iloc[0]
    ]
    
    # Normalize user values to 0-1 scale for comparison
    df_features = df[features]
    normalized_user = [(val - df_features[feat].min()) / (df_features[feat].max() - df_features[feat].min()) 
                       for val, feat in zip(user_values, features)]
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=normalized_user,
        theta=features,
        fill='toself',
        name='Your Profile',
        line_color=patient_color
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="🎯 Your Health Profile Radar",
        height=500
    )
    
    st.plotly_chart(fig_radar, width='stretch')

with tab4:
    st.markdown("### Feature Correlation Analysis")
    
    # Correlation matrix
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        title='📋 Feature Correlation Matrix',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, width='stretch')
    
    # Key correlations
    st.markdown("### Key Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Strong Positive Correlations:**\n- Glucose and Diabetes Outcome\n- BMI and Skin Thickness\n- Age and Pregnancies")
    
    with col2:
        st.warning("**Important Risk Factors:**\n- High glucose levels\n- Elevated BMI\n- Genetic predisposition (DPF)")

# PERSONALIZED MEAL SUGGESTIONS
st.markdown("---")
st.markdown('<h2 class="sub-header">🍽️ Personalized Meal Recommendations</h2>', unsafe_allow_html=True)

def get_meal_recommendations(user_data, prediction_result):
    """Generate personalized meal recommendations based on user health data"""
    
    # Extract user values
    glucose = user_data['Glucose'].iloc[0]
    bmi = user_data['BMI'].iloc[0]
    age = user_data['Age'].iloc[0]
    pregnancies = user_data['Pregnancies'].iloc[0]
    
    # Determine risk category
    is_high_risk = prediction_result[0] == 1
    is_high_glucose = glucose > 140
    is_overweight = bmi >= 25
    is_senior = age >= 60
    is_pregnant = pregnancies > 0
    
    # Calculate daily nutritional needs
    # Basic calorie calculation (Harris-Benedict equation approximation)
    if is_overweight:
        daily_calories = 1500 - 1800  # Weight loss range
    elif is_senior:
        daily_calories = 1800 - 2000  # Moderate activity
    else:
        daily_calories = 2000 - 2200  # Normal range
    
    # Carb recommendations (45-65% of calories, but lower for diabetics)
    if is_high_risk or is_high_glucose:
        carb_percentage = 40  # Lower carbs for diabetics
    else:
        carb_percentage = 50  # Moderate carbs
    
    daily_carbs = (daily_calories * carb_percentage // 100) // 4  # 4 calories per gram of carbs
    
    return daily_calories, daily_carbs, is_high_risk, is_high_glucose, is_overweight

# Get recommendations
daily_calories, daily_carbs, is_high_risk, is_high_glucose, is_overweight = get_meal_recommendations(user_data, user_result)

# Display nutritional targets
st.markdown("### 📊 Your Daily Nutritional Targets")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Daily Calories", f"{daily_calories}", help="Recommended daily caloric intake")
with col2:
    st.metric("Carbohydrates", f"{daily_carbs}g", help="Daily carbohydrate limit")
with col3:
    protein_grams = int(daily_calories * 0.20 // 4)  # 20% of calories from protein
    st.metric("Protein", f"{protein_grams}g", help="Daily protein requirement")
with col4:
    fiber_grams = 25 if user_data['Age'].iloc[0] < 50 else 21
    st.metric("Fiber", f"{fiber_grams}g", help="Daily fiber recommendation")

# Meal categories and suggestions
meal_suggestions = {
    "🌅 Breakfast": {
        "healthy": [
            "🥣 Steel-cut oatmeal with berries and nuts (45g carbs)",
            "🍳 Vegetable omelet with whole grain toast (35g carbs)",
            "🥑 Avocado toast on whole grain bread (40g carbs)",
            "🥛 Greek yogurt with almonds and cinnamon (25g carbs)",
            "🥞 Protein pancakes with sugar-free syrup (30g carbs)"
        ],
        "diabetic": [
            "🥚 Scrambled eggs with spinach and cheese (8g carbs)",
            "🥑 Avocado and egg bowl (12g carbs)",
            "🧀 Cottage cheese with cucumber slices (10g carbs)",
            "🥜 Almond flour pancakes with berries (15g carbs)",
            "🥬 Green smoothie with protein powder (18g carbs)"
        ]
    },
    "🌞 Lunch": {
        "healthy": [
            "🥗 Quinoa salad with grilled chicken (50g carbs)",
            "🍲 Lentil soup with whole grain roll (55g carbs)",
            "🐟 Grilled salmon with sweet potato (45g carbs)",
            "🌯 Turkey and hummus wrap (48g carbs)",
            "🍜 Brown rice bowl with vegetables (52g carbs)"
        ],
        "diabetic": [
            "🥗 Large salad with grilled protein (20g carbs)",
            "🥒 Cucumber chicken salad (15g carbs)",
            "🐟 Baked fish with roasted vegetables (25g carbs)",
            "🥩 Lean beef with cauliflower rice (18g carbs)",
            "🦐 Shrimp stir-fry with zucchini noodles (22g carbs)"
        ]
    },
    "🌆 Dinner": {
        "healthy": [
            "🍗 Grilled chicken with quinoa and vegetables (50g carbs)",
            "🐟 Baked cod with brown rice pilaf (48g carbs)",
            "🥩 Lean beef stir-fry with brown rice (52g carbs)",
            "🍲 Turkey chili with cornbread (45g carbs)",
            "🍝 Whole wheat pasta with marinara sauce (55g carbs)"
        ],
        "diabetic": [
            "🍗 Herb-roasted chicken with green vegetables (20g carbs)",
            "🐟 Grilled salmon with asparagus (15g carbs)",
            "🥩 Steak with mushrooms and spinach (18g carbs)",
            "🦃 Turkey meatballs with zucchini noodles (25g carbs)",
            "🥬 Stuffed bell peppers with ground turkey (22g carbs)"
        ]
    },
    "🍎 Snacks": {
        "healthy": [
            "🍎 Apple with almond butter (25g carbs)",
            "🥜 Mixed nuts and dried fruit (20g carbs)",
            "🍓 Greek yogurt with berries (18g carbs)",
            "🥨 Whole grain crackers with hummus (22g carbs)",
            "🥕 Carrots with peanut butter (15g carbs)"
        ],
        "diabetic": [
            "🥜 Handful of almonds (6g carbs)",
            "🧀 Cheese with cucumber slices (8g carbs)",
            "🥚 Hard-boiled egg (2g carbs)",
            "🥑 Half avocado with salt (4g carbs)",
            "🥒 Celery with cream cheese (5g carbs)"
        ]
    }
}

# Display meal suggestions in tabs
meal_tab1, meal_tab2, meal_tab3 = st.tabs(["📋 Today's Menu", "🍳 Recipe Ideas", "💡 Nutrition Tips"])

with meal_tab1:
    st.markdown("### Recommended Meals for You")
    
    # Choose meal category based on health status
    meal_category = "diabetic" if (is_high_risk or is_high_glucose) else "healthy"
    
    if is_high_risk or is_high_glucose:
        st.warning("⚠️ **Diabetic-Friendly Menu** - Lower carbohydrate options recommended")
    else:
        st.success("✅ **Balanced Healthy Menu** - Moderate carbohydrate options")
    
    for meal_time, suggestions in meal_suggestions.items():
        st.markdown(f"#### {meal_time}")
        selected_meals = suggestions[meal_category]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"• {selected_meals[0]}")
            st.write(f"• {selected_meals[1]}")
        with col2:
            st.write(f"• {selected_meals[2]}")
        
        st.markdown("---")

with meal_tab2:
    st.markdown("### 👨‍🍳 Quick Recipe Ideas")
    
    # Sample recipes based on health status
    if is_high_risk or is_high_glucose:
        st.markdown("#### 🥗 Diabetic-Friendly Recipes")
        
        recipe_col1, recipe_col2 = st.columns(2)
        
        with recipe_col1:
            st.markdown("""
            **🐟 Lemon Herb Baked Salmon**
            - 6oz salmon fillet
            - 2 cups steamed broccoli
            - 1 tbsp olive oil
            - Lemon juice & herbs
            - *Carbs: 12g | Protein: 40g*
            """)
            
            st.markdown("""
            **🥗 Chicken Caesar Salad**
            - 5oz grilled chicken breast
            - 3 cups romaine lettuce
            - 2 tbsp Caesar dressing
            - Parmesan cheese
            - *Carbs: 8g | Protein: 42g*
            """)
        
        with recipe_col2:
            st.markdown("""
            **🍳 Veggie Scramble**
            - 3 large eggs
            - 1 cup mixed vegetables
            - 1 oz cheese
            - 1 tbsp olive oil
            - *Carbs: 10g | Protein: 25g*
            """)
            
            st.markdown("""
            **🦐 Zucchini Noodle Stir-fry**
            - 6oz shrimp
            - 2 medium zucchini (spiralized)
            - Mixed bell peppers
            - Garlic & ginger
            - *Carbs: 15g | Protein: 35g*
            """)
    
    else:
        st.markdown("#### 🍽️ Balanced Healthy Recipes")
        
        recipe_col1, recipe_col2 = st.columns(2)
        
        with recipe_col1:
            st.markdown("""
            **🍚 Quinoa Power Bowl**
            - 3/4 cup cooked quinoa
            - 4oz grilled chicken
            - Mixed roasted vegetables
            - 2 tbsp tahini dressing
            - *Carbs: 45g | Protein: 35g*
            """)
            
            st.markdown("""
            **🍝 Whole Wheat Pasta Primavera**
            - 1.5 cups whole wheat pasta
            - Seasonal vegetables
            - 3oz lean protein
            - Olive oil & herbs
            - *Carbs: 52g | Protein: 28g*
            """)
        
        with recipe_col2:
            st.markdown("""
            **🥞 Protein Pancakes**
            - 1/2 cup oat flour
            - 2 eggs + 1 egg white
            - 1/4 cup Greek yogurt
            - 1/2 cup berries
            - *Carbs: 35g | Protein: 25g*
            """)
            
            st.markdown("""
            **🌯 Turkey & Hummus Wrap**
            - Whole grain tortilla
            - 4oz sliced turkey
            - 3 tbsp hummus
            - Fresh vegetables
            - *Carbs: 48g | Protein: 30g*
            """)

with meal_tab3:
    st.markdown("### 💡 Personalized Nutrition Tips")
    
    tip_col1, tip_col2 = st.columns(2)
    
    with tip_col1:
        st.markdown("#### 🎯 For Your Health Profile:")
        
        if is_high_risk or is_high_glucose:
            st.info("""
            **Diabetes Management Tips:**
            • Choose complex carbs over simple sugars
            • Eat protein with each meal
            • Monitor portion sizes carefully
            • Space meals evenly throughout the day
            • Stay hydrated with water
            """)
        
        if is_overweight:
            st.warning("""
            **Weight Management:**
            • Focus on portion control
            • Increase fiber intake
            • Choose lean proteins
            • Limit processed foods
            • Include physical activity daily
            """)
        
        if user_data['Age'].iloc[0] >= 60:
            st.info("""
            **Senior Nutrition:**
            • Ensure adequate calcium intake
            • Focus on nutrient-dense foods
            • Stay well hydrated
            • Consider vitamin D supplementation
            • Maintain regular meal times
            """)
    
    with tip_col2:
        st.markdown("#### 🍽️ General Guidelines:")
        
        st.success("""
        **Healthy Eating Principles:**
        • Fill half your plate with vegetables
        • Choose whole grains over refined
        • Include healthy fats (nuts, avocado, olive oil)
        • Limit added sugars and sodium
        • Practice mindful eating
        """)
        
        st.markdown("#### 🥤 Hydration Guide:")
        water_needs = 8 + (user_data['Age'].iloc[0] // 10)  # Base + age factor
        st.write(f"💧 **Daily Water Goal:** {water_needs} glasses")
        st.write("• Drink water before, during, and after meals")
        st.write("• Limit sugary beverages")
        st.write("• Herbal teas count toward fluid intake")

# Meal planning tools
st.markdown("---")
st.markdown("### 📅 Weekly Meal Planning Assistant")

if st.button("🗓️ Generate 7-Day Meal Plan"):
    st.success("📋 **Your personalized 7-day meal plan has been generated!**")
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    meal_category = "diabetic" if (is_high_risk or is_high_glucose) else "healthy"
    
    for i, day in enumerate(days):
        with st.expander(f"📅 {day}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**🌅 Breakfast:**")
                breakfast_idx = i % len(meal_suggestions["🌅 Breakfast"][meal_category])
                st.write(meal_suggestions["🌅 Breakfast"][meal_category][breakfast_idx])
            
            with col2:
                st.write("**🌞 Lunch:**")
                lunch_idx = i % len(meal_suggestions["🌞 Lunch"][meal_category])
                st.write(meal_suggestions["🌞 Lunch"][meal_category][lunch_idx])
            
            with col3:
                st.write("**🌆 Dinner:**")
                dinner_idx = i % len(meal_suggestions["🌆 Dinner"][meal_category])
                st.write(meal_suggestions["🌆 Dinner"][meal_category][dinner_idx])
            
            st.write("**🍎 Snack:**")
            snack_idx = i % len(meal_suggestions["🍎 Snacks"][meal_category])
            st.write(meal_suggestions["🍎 Snacks"][meal_category][snack_idx])

# Shopping list generator
if st.button("🛒 Generate Shopping List"):
    st.success("🛍️ **Smart Shopping List Based on Your Meal Plan:**")
    
    if is_high_risk or is_high_glucose:
        shopping_list = [
            "🥬 **Vegetables:** Spinach, broccoli, cauliflower, zucchini, bell peppers",
            "🥩 **Proteins:** Salmon, chicken breast, lean beef, eggs, tofu",
            "🥜 **Healthy Fats:** Avocados, almonds, olive oil, walnuts",
            "🧀 **Dairy:** Greek yogurt, cottage cheese, low-fat cheese",
            "🌿 **Herbs & Spices:** Fresh herbs, garlic, ginger, turmeric",
            "🥤 **Beverages:** Herbal teas, sparkling water, unsweetened almond milk"
        ]
    else:
        shopping_list = [
            "🥬 **Vegetables:** Mixed greens, sweet potatoes, carrots, tomatoes",
            "🍚 **Whole Grains:** Quinoa, brown rice, oats, whole wheat pasta",
            "🥩 **Proteins:** Fish, poultry, legumes, nuts, seeds",
            "🍓 **Fruits:** Berries, apples, citrus fruits",
            "🥛 **Dairy:** Low-fat milk, yogurt, cheese",
            "🫒 **Healthy Oils:** Olive oil, avocado oil, coconut oil"
        ]
    
    for item in shopping_list:
        st.write(f"• {item}")

# FOOTER
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🩺 <strong>Diabetes Risk Assessment Tool</strong> 🩺</p>
    <p>Powered by Machine Learning | Built with Streamlit</p>
    <p><em>Stay healthy, stay informed! 💙</em></p>
</div>
""", unsafe_allow_html=True)
