import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
try:
    from streamlit_lottie import st_lottie
except Exception:
    # keep a fallback so missing optional dependency doesn't crash the whole app
    st_lottie = None
try:
    from streamlit_extras.switch_page_button import switch_page
except Exception:
    # provide a non-crashing fallback so the app can still run without this optional helper
    def switch_page(page_name: str):
        # store the requested page in session_state so users can manually navigate
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
            # ignore if Streamlit UI isn't available yet
            pass

BASE_DIR = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Diabetes Risk — Home", layout="wide", page_icon="🩺")

# Inform the user (non-blocking) if the optional lottie package is missing
if st_lottie is None:
    try:
        # sidebar is a safe place for notices that don't interrupt layout
        st.sidebar.warning("Optional package 'streamlit-lottie' is not installed. Install it with `pip install streamlit-lottie` to enable animations.")
    except Exception:
        # if Streamlit runtime isn't ready yet, ignore
        pass

st.markdown("""
<style>
/* soft blue theme */
.header {text-align:center; color:#2E86AB; font-size:34px; font-weight:700}
.card {background:#fff; padding:18px; border-radius:12px; box-shadow:0 6px 18px rgba(46,134,171,0.08)}
.muted {color:#6b7280}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">🩺 Diabetes Risk Assessment — Patient Info</div>', unsafe_allow_html=True)

# Add a subtle animated 3D background using three.js embedded via an HTML component.
import streamlit.components.v1 as components

def _threejs_header(height=220):
        html = f"""
        <div id='threejs' style='width:100%;height:{height}px;position:relative;overflow:hidden;border-radius:12px'></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
        <script>
        (function(){{
            const container = document.getElementById('threejs');
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(50, container.clientWidth / {height}, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({{alpha:true, antialias:true}});
            renderer.setSize(container.clientWidth, {height});
            container.appendChild(renderer.domElement);

            camera.position.z = 3.5;

            const geom = new THREE.IcosahedronGeometry(1.1, 3);
            const mat = new THREE.MeshStandardMaterial({{color:0x2e86ab, metalness:0.2, roughness:0.6, transparent:true, opacity:0.95}});
            const mesh = new THREE.Mesh(geom, mat);
            scene.add(mesh);

            // soft lights
            const light1 = new THREE.DirectionalLight(0xffffff, 0.8);
            light1.position.set(5,5,5);
            scene.add(light1);
            const light2 = new THREE.AmbientLight(0xffffff, 0.4);
            scene.add(light2);

            function onResize(){{
                renderer.setSize(container.clientWidth, {height});
                camera.aspect = container.clientWidth / {height};
                camera.updateProjectionMatrix();
            }}
            window.addEventListener('resize', onResize);

            function animate(){{
                requestAnimationFrame(animate);
                mesh.rotation.x += 0.008;
                mesh.rotation.y += 0.012;
                mesh.rotation.z += 0.006;
                renderer.render(scene, camera);
            }}
            animate();
        }})();
        </script>
        """
        return html

# Render the 3D header (non-blocking)
try:
        components.html(_threejs_header(), height=220)
except Exception:
        # If components rendering fails, continue without crashing
        pass

cols = st.columns([1, 2, 1])
with cols[1]:
    st.markdown("""
    <div class='card'>
    <p class='muted'>Please enter the patient's health metrics below. Use the button to predict risk and move to the results page.</p>
    </div>
    """, unsafe_allow_html=True)

df_path = BASE_DIR / "diabetes.csv"
if df_path.exists():
    df = pd.read_csv(df_path)
else:
    df = pd.DataFrame()

def get_defaults():
    # pick medians from dataset if available
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

with st.form("patient_form"):
    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input('🤰 Pregnancies', min_value=0, max_value=20, value=defaults['Pregnancies'], step=1)
        glucose = st.slider('🍬 Glucose (mg/dL)', min_value=0, max_value=300, value=defaults['Glucose'])
        bp = st.slider('💓 Blood Pressure (mmHg)', min_value=0, max_value=200, value=defaults['BloodPressure'])
        skin = st.slider('📏 Skin Thickness (mm)', min_value=0, max_value=100, value=defaults['SkinThickness'])
    with c2:
        insulin = st.slider('💉 Insulin (μU/mL)', min_value=0, max_value=1000, value=defaults['Insulin'])
        bmi = st.number_input('⚖️ BMI (kg/m²)', min_value=0.0, max_value=100.0, value=float(defaults['BMI']), step=0.1, format="%.1f")
        dpf = st.number_input('🧬 Diabetes Pedigree Function', min_value=0.0, max_value=5.0, value=float(defaults['DiabetesPedigreeFunction']), step=0.01, format="%.2f")
        age = st.number_input('🎂 Age', min_value=0, max_value=120, value=defaults['Age'])

    st.markdown("---")
    # progress indicator for completeness
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
    st.progress(completeness)

    submitted = st.form_submit_button("🔍 Predict Risk")

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
    # navigate to prediction page
    try:
        switch_page('1_Prediction_Result')
    except Exception:
        # fallback: inform user to use sidebar pages
        st.success("Inputs saved. Please open the Prediction Result page from the sidebar.")

st.markdown("---")
with st.expander("📈 Dataset overview", expanded=False):
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", len(df))
        col2.metric("Diabetic Cases", int(df['Outcome'].sum()))
        col3.metric("Diabetes Rate", f"{df['Outcome'].mean()*100:.1f}%")
    else:
        st.info("No dataset found in the project root. Place `diabetes.csv` alongside this app for dataset visuals.")
