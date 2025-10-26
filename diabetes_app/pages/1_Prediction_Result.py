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
import streamlit.components.v1 as components
try:
    from streamlit_extras.switch_page_button import switch_page
except Exception:
    # non-fatal fallback when package is not installed
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

st.set_page_config(page_title="Prediction Result", layout='wide', page_icon='📊')

st.markdown("""
<style>
.result-header{font-size:28px;color:#2E86AB;font-weight:700}
.result-card{background:#fff;padding:18px;border-radius:12px;box-shadow:0 8px 20px rgba(0,0,0,0.06)}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="result-header">📊 Prediction Result</div>', unsafe_allow_html=True)

# small three.js vignette near the result for visual polish
def _mini_threejs(size=200):
    html = f"""
    <div id='mini' style='width:100%;height:{size}px'></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
    <script>
    (function(){{
      const container = document.getElementById('mini');
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(50, container.clientWidth / {size}, 0.1, 1000);
      const renderer = new THREE.WebGLRenderer({{alpha:true, antialias:true}});
      renderer.setSize(container.clientWidth, {size});
      container.appendChild(renderer.domElement);
      camera.position.z = 2.5;
      const geom = new THREE.TorusKnotGeometry(0.6, 0.2, 100, 16);
      const mat = new THREE.MeshStandardMaterial({{color:0xf5576c, metalness:0.3, roughness:0.6}});
      const mesh = new THREE.Mesh(geom, mat);
      scene.add(mesh);
      const light = new THREE.DirectionalLight(0xffffff,1);
      light.position.set(5,5,5);
      scene.add(light);
      function animate(){{requestAnimationFrame(animate); mesh.rotation.x += 0.01; mesh.rotation.y += 0.013; renderer.render(scene,camera); }}
      animate();
    }})();
    </script>
    """
    return html

try:
    components.html(_mini_threejs(200), height=200)
except Exception:
    pass

# get input from session
if 'patient_input' not in st.session_state:
    st.warning('No patient data found. Please provide patient info on the Home page.')
    st.stop()

input_data = st.session_state['patient_input']
df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()

def ensure_model():
    # if model exists, load; else train and save
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        return model, None

    # train model quickly and persist
    if df.empty:
        st.error('No dataset available to train the fallback model. Add diabetes.csv to project root.')
        st.stop()

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # save
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

# show donut chart
healthy_prob = prob[0]*100
diabetic_prob = prob[1]*100

col1, col2 = st.columns([1,2])
with col1:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    if prediction == 0:
        st.success('✅ Low Risk — Keep up your healthy habits!')
    else:
        st.warning('⚠️ High Risk — Consult your doctor soon.')
    st.markdown(f"**Probability:** {max(healthy_prob, diabetic_prob):.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    fig = go.Figure(data=[go.Pie(labels=['Healthy', 'Diabetic'], values=[healthy_prob, diabetic_prob], hole=0.6,
                                marker_colors=['#2E86AB', '#E74C3C'])])
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(title='Prediction Confidence', showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

st.markdown('---')

if train_accuracy is not None:
    st.info(f"Model was trained on-the-fly. Test accuracy: {train_accuracy*100:.1f}%")

st.markdown('### Patient vs Population')
if not df.empty:
    import plotly.express as px
    fig = px.scatter(df, x='Age', y='Glucose', color='Outcome', color_discrete_map={0:'#2E86AB',1:'#E74C3C'}, opacity=0.6)
    fig.add_scatter(x=[input_df['Age'].iloc[0]], y=[input_df['Glucose'].iloc[0]], mode='markers', marker=dict(size=14,color='#000000'), name='You')
    st.plotly_chart(fig, use_container_width=True)

st.markdown('---')
cols = st.columns(3)
cols[0].metric('Healthy Probability', f"{healthy_prob:.1f}%")
cols[1].metric('Diabetic Probability', f"{diabetic_prob:.1f}%")
if train_accuracy is not None:
    cols[2].metric('Model Accuracy (test)', f"{train_accuracy*100:.1f}%")

st.markdown('---')
if st.button('🍎 View Personalized Diet Plan'):
    try:
        switch_page('2_Diet_Recommendations')
    except Exception:
        st.success('Prediction saved. Please open the Diet Recommendations page from the sidebar.')
