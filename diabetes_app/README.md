# Diabetes Risk Assessment — Streamlit Multipage App

This folder contains a 3-page Streamlit application that collects patient information, predicts diabetes risk using a RandomForest model, and provides personalized diet recommendations.

How to run:

1. Ensure your working directory is the project root (where `diabetes.csv` is located).
2. (Optional) Create a virtualenv and install dependencies:

   pip install -r diabetes_app/requirements.txt

3. Run the app (point to Home.py):

   streamlit run diabetes_app/Home.py

Notes:
- If `diabetes_app/model.pkl` is missing, the app will train a RandomForest on `diabetes.csv` and save `model.pkl` automatically.
- Place `diabetes.csv` in the project root (already present in repo) to enable dataset visuals and model training.

Files:
- `Home.py` — Patient information page and dataset overview
- `pages/1_Prediction_Result.py` — Prediction, charts and probability visualization
- `pages/2_Diet_Recommendations.py` — Personalized diet recommendations and goals
- `requirements.txt` — App-specific python dependencies

Next steps / enhancements (optional):
- Add Lottie JSON files to `assets/` and use `st_lottie` to display animations.
- Add SHAP explainability visuals for per-patient explanations.
- Improve navigation with animated transitions and mobile responsive tweaks.
