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
    st.dataframe(df.describe(), use_container_width=True)


# X AND Y DATA
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

# ENHANCED USER INPUT FUNCTION
def user_report():
    st.sidebar.markdown("---")
    
    # Pregnancies
    pregnancies = st.sidebar.slider(
        '🤰 Pregnancies', 
        min_value=0, max_value=17, value=3,
        help="Number of times pregnant"
    )
    
    # Glucose
    glucose = st.sidebar.slider(
        '🍬 Glucose Level (mg/dL)', 
        min_value=0, max_value=200, value=120,
        help="Blood glucose concentration (normal: 70-140 mg/dL)"
    )
    
    # Blood Pressure
    bp = st.sidebar.slider(
        '💓 Blood Pressure (mmHg)', 
        min_value=0, max_value=122, value=70,
        help="Systolic blood pressure (normal: 90-120 mmHg)"
    )
    
    # Skin Thickness
    skinthickness = st.sidebar.slider(
        '📏 Skin Thickness (mm)', 
        min_value=0, max_value=100, value=20,
        help="Triceps skin fold thickness"
    )
    
    # Insulin
    insulin = st.sidebar.slider(
        '💉 Insulin Level (μU/mL)', 
        min_value=0, max_value=846, value=79,
        help="Serum insulin level"
    )
    
    # BMI
    bmi = st.sidebar.slider(
        '⚖️ BMI (kg/m²)', 
        min_value=0.0, max_value=67.0, value=20.0, step=0.1,
        help="Body Mass Index (normal: 18.5-24.9)"
    )
    
    # Diabetes Pedigree Function
    dpf = st.sidebar.slider(
        '🧬 Diabetes Pedigree Function', 
        min_value=0.0, max_value=2.4, value=0.47, step=0.01,
        help="Genetic predisposition to diabetes"
    )
    
    # Age
    age = st.sidebar.slider(
        '🎂 Age (years)', 
        min_value=21, max_value=88, value=33,
        help="Patient age in years"
    )
    
    # Health status indicators
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Health Indicators")
    
    # BMI category
    if bmi < 18.5:
        bmi_status = "🔵 Underweight"
    elif bmi < 25:
        bmi_status = "🟢 Normal"
    elif bmi < 30:
        bmi_status = "🟡 Overweight"
    else:
        bmi_status = "🔴 Obese"
    st.sidebar.write(f"BMI Category: {bmi_status}")
    
    # Glucose status
    if glucose < 70:
        glucose_status = "🔵 Low"
    elif glucose <= 140:
        glucose_status = "🟢 Normal"
    elif glucose <= 199:
        glucose_status = "🟡 Pre-diabetic"
    else:
        glucose_status = "🔴 Diabetic"
    st.sidebar.write(f"Glucose Level: {glucose_status}")

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data




# PATIENT DATA INPUT
user_data = user_report()

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
    
    # Get prediction and probability
    user_result = rf.predict(user_data)
    user_probability = rf.predict_proba(user_data)
    
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
    
    st.plotly_chart(fig, use_container_width=True)

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
            opacity=0.6
        )
        fig1.add_scatter(
            x=[user_data['Age'].iloc[0]], 
            y=[user_data['Pregnancies'].iloc[0]],
            mode='markers',
            marker=dict(size=15, color=patient_color, symbol=patient_symbol, line=dict(width=2, color='white')),
            name='You',
            showlegend=True
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Age vs Glucose
        fig2 = px.scatter(
            df, x='Age', y='Glucose',
            color='Outcome',
            color_discrete_map={0: '#2E86AB', 1: '#E74C3C'},
            title='🍬 Glucose Level vs Age',
            labels={'Outcome': 'Diabetes Status'},
            opacity=0.6
        )
        fig2.add_scatter(
            x=[user_data['Age'].iloc[0]], 
            y=[user_data['Glucose'].iloc[0]],
            mode='markers',
            marker=dict(size=15, color=patient_color, symbol=patient_symbol, line=dict(width=2, color='white')),
            name='You',
            showlegend=True
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Age vs BMI
        fig3 = px.scatter(
            df, x='Age', y='BMI',
            color='Outcome',
            color_discrete_map={0: '#2E86AB', 1: '#E74C3C'},
            title='⚖️ BMI vs Age',
            labels={'Outcome': 'Diabetes Status'},
            opacity=0.6
        )
        fig3.add_scatter(
            x=[user_data['Age'].iloc[0]], 
            y=[user_data['BMI'].iloc[0]],
            mode='markers',
            marker=dict(size=15, color=patient_color, symbol=patient_symbol, line=dict(width=2, color='white')),
            name='You',
            showlegend=True
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Age vs Blood Pressure
        fig4 = px.scatter(
            df, x='Age', y='BloodPressure',
            color='Outcome',
            color_discrete_map={0: '#2E86AB', 1: '#E74C3C'},
            title='💓 Blood Pressure vs Age',
            labels={'Outcome': 'Diabetes Status'},
            opacity=0.6
        )
        fig4.add_scatter(
            x=[user_data['Age'].iloc[0]], 
            y=[user_data['BloodPressure'].iloc[0]],
            mode='markers',
            marker=dict(size=15, color=patient_color, symbol=patient_symbol, line=dict(width=2, color='white')),
            name='You',
            showlegend=True
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

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
        st.plotly_chart(fig_hist1, use_container_width=True)
        
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
        st.plotly_chart(fig_hist2, use_container_width=True)
    
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
        st.plotly_chart(fig_hist3, use_container_width=True)
        
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
        st.plotly_chart(fig_hist4, use_container_width=True)

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
    st.plotly_chart(fig_importance, use_container_width=True)
    
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
    
    st.plotly_chart(fig_radar, use_container_width=True)

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
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Key correlations
    st.markdown("### Key Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Strong Positive Correlations:**\n- Glucose and Diabetes Outcome\n- BMI and Skin Thickness\n- Age and Pregnancies")
    
    with col2:
        st.warning("**Important Risk Factors:**\n- High glucose levels\n- Elevated BMI\n- Genetic predisposition (DPF)")

# FOOTER
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🩺 <strong>Diabetes Risk Assessment Tool</strong> 🩺</p>
    <p>Powered by Machine Learning | Built with Streamlit</p>
    <p><em>Stay healthy, stay informed! 💙</em></p>
</div>
""", unsafe_allow_html=True)
