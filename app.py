import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64
import os
import shap
import matplotlib.pyplot as plt

# ===========================
#     FILE PATHS & SETUP
# ===========================
IMAGE_PATH = "img/مشروم.png"
MODEL_PATH = "mushroom_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"
DATA_PATH = "mushroom.csv"  # REQUIRED for SHAP background data


# ===========================
#     IMAGE HANDLING
# ===========================
def get_base64_of_file(path):
    if not os.path.exists(path):
        st.warning(f"Warning: Background image not found at {path}. Using fallback background.")
        return None
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


image_base64 = get_base64_of_file(IMAGE_PATH)

# ===========================
#     CONSTANTS & LOAD MODEL & SHAP (Run Once)
# ===========================

# --- Initialize Session State ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False

try:
    # Load core components
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)

    # List of all features the model expects (for validation later)
    ALL_MODEL_FEATURES = list(label_encoders.keys())

    # List of features for UI input
    features = list(label_encoders.keys())
    if 'class' in features:
        features.remove('class')
    if 'veil-type' in features:
        features.remove('veil-type')  # Typically a constant value

    # --- SHAP Setup (Crucial for performance, runs once) ---
    try:
        df_full = pd.read_csv(DATA_PATH)
        # Use a small background sample for faster SHAP calculation if TreeExplainer is not used
        df_sample = df_full.drop("class", axis=1).iloc[:100]

        
        for col in df_sample.columns:
            df_sample[col] = pd.to_numeric(
                df_sample[col].astype(str).apply(lambda x: ord(x) if len(x) == 1 else x),
                errors='coerce'
            ).fillna(0).astype(int)


        # 1. Try TreeExplainer (fastest for tree models like RandomForest, XGBoost)
        try:
            explainer = shap.TreeExplainer(model)
            st.session_state['explainer'] = explainer
            st.session_state['explainer_type'] = 'tree'
            st.session_state['shap_background'] = None
            st.write("✅ SHAP TreeExplainer loaded successfully.")
        except Exception:
            # 2. Fallback to General Explainer (slower, requires data)
            explainer = shap.Explainer(model.predict, df_sample)
            st.session_state['explainer'] = explainer
            st.session_state['explainer_type'] = 'general'
            st.session_state['shap_background'] = df_sample
            st.write("⚠ SHAP General Explainer loaded (may be slower).")

    except FileNotFoundError:
        st.error("Error: mushroom.csv not found. SHAP explanation feature will not work.")
        st.session_state['explainer'] = None

    st.session_state['model_loaded'] = True

except FileNotFoundError as e:
    st.error(f"Error: Model files not found. Please ensure '{MODEL_PATH}' and '{ENCODERS_PATH}' are correct. {e}")
    st.stop()

# ===========================
#     PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="Mushroom Classifier",
    layout="wide"
)

# ===========================
#     HIDE STREAMLIT HEADER/MENU/FOOTER
# ===========================
st.markdown("""
    <style>
        /* Hide Streamlit header, hamburger menu, and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Mapping of feature values
MUSHROOM_MAPPING = {
    "cap-shape": ["bell (b)", "conical (s)", "convex (x)", "flat (f)", "knobbed (k)", "sunken (c)"],
    "cap-surface": ["fibrous (f)", "grooves (g)", "scaly (y)", "smooth (s)"],
    "cap-color": ["brown (n)", "buff (b)", "cinnamon (c)", "gray (g)", "green (r)", "pink (p)",
                  "purple (u)", "red (e)", "white (w)", "yellow (y)"],
    "bruises": ["yes (t)", "no (f)"],
    "odor": ["almond (a)", "anise (l)", "creosote (c)", "fishy (y)", "foul (f)", "musty (m)",
             "none (n)", "pungent (p)", "spicy (s)"],
    "gill-attachment": ["attached (a)", "free (f)"],
    "gill-spacing": ["close (c)", "crowded (w)"],
    "gill-size": ["broad (b)", "narrow (n)"],
    "gill-color": ["black (k)", "brown (n)", "buff (b)", "chocolate (h)", "gray (g)", "green (r)",
                   "orange (o)", "pink (p)", "purple (u)", "red (e)", "white (w)", "yellow (y)"],
    "stalk-shape": ["enlarging (e)", "tapering (t)"],
    "stalk-root": ["bulbous (b)", "club (c)", "equal (e)", "rooted (r)", "missing (?)"],
    "stalk-surface-above-ring": ["fibrous (f)", "scaly (y)", "silky (k)", "smooth (s)"],
    "stalk-surface-below-ring": ["fibrous (f)", "scaly (y)", "silky (k)", "smooth (s)"],
    "stalk-color-above-ring": ["brown (n)", "buff (b)", "cinnamon (c)", "gray (g)", "orange (o)",
                               "pink (p)", "red (e)", "white (w)", "yellow (y)"],
    "stalk-color-below-ring": ["brown (n)", "buff (b)", "cinnamon (c)", "gray (g)", "orange (o)",
                               "pink (p)", "red (e)", "white (w)", "yellow (y)"],
    "veil-color": ["brown (n)", "orange (o)", "white (w)", "yellow (y)"],
    "ring-number": ["none (n)", "one (o)", "two (t)"],
    "ring-type": ["evanescent (e)", "flaring (f)", "large (l)", "none (n)", "pendant (p)"],
    "spore-print-color": ["black (k)", "brown (n)", "buff (b)", "chocolate (h)", "green (r)",
                          "orange (o)", "purple (u)", "white (w)", "yellow (y)"],
    "population": ["abundant (a)", "clustered (c)", "numerous (n)", "scattered (s)",
                   "several (v)", "solitary (y)"],
    "habitat": ["grasses (g)", "leaves (l)", "meadows (m)", "paths (p)",
                "urban (u)", "waste (w)", "woods (d)"]
}

# ===========================
#     CUSTOM CSS (Improved Font & Clarity)
# ===========================

background_style = ""
if image_base64:
    background_style = f"""
    .stApp {{
        background: url(data:image/png;base64,{image_base64}) no-repeat center center fixed;
        background-size: cover;
    }}
    """
else:
    background_style = ".stApp { background-color: #0b111a; }"

st.markdown(f"""
<style>
    {background_style}

    /* Styling for the main container to create a transparent/dark overlay */
    .main > div {{
        background: rgba(0, 0, 0, 0.65); /* Dark transparent layer */
        padding: 30px;
        border-radius: 10px;
        backdrop-filter: blur(4px); /* Light blur effect */
    }}

    /* Page Titles (h1) */
    h1 {{
        color: #e59866; /* Gold/Orange color for contrast */
        text-align: center;
        font-weight: 900;
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 20px;
    }}

    /* Headers (h2, h3) */
    h2, h3, .st-emotion-cache-1wq0w3v {{ /* Added Streamlit default header class */
        color: #f0f3f4; /* Light text for headers */
        font-weight: 700;
        text-shadow: 1px 1px 2px #000000;
    }}

    /* Input labels and general text (Increased font size and brightness) */
    .stSelectbox label, .stTextInput label, .stMarkdown, p, li {{
        font-weight: 600;
        color: #f0f3f4; /* Increased brightness for better visibility */
        font-size: 18px; 
    }}

    /* Selectbox styling */
    div[data-baseweb="select"] div[role="button"] {{
        background-color: #1c2833 !important; /* Very dark background */
        color: #f0f3f4 !important; /* Light text color */
        border: 1px solid #5d6d7e;
    }}

    /* Navigation/Predict Button */
    .stButton>button {{
        background-color: #5d6d7e !important;
        color: white !important;
        padding: 0.7rem 1.2rem;
        font-size: 18px;
        border-radius: 10px;
        width: 100%;
        margin-top: 15px;
        border: 2px solid #aeb6bf; 
        transition: all 0.2s;
    }}
    .stButton>button:hover {{
        background-color: #85929e !important;
        border-color: #f7f9f9;
        transform: translateY(-2px);
    }}

    /* Result Box styling in Explanation Page */
    .stSuccess, .stError {{
        background-color: rgba(23, 32, 42, 0.9);
        color: white;
        border-left: 5px solid;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }}
    .stSuccess {{ border-left-color: #2ecc71; }} /* Green for Edible */
    .stError {{ border-left-color: #e74c3c; }} /* Red for Poisonous */
</style>
""", unsafe_allow_html=True)


# ===========================
#     HOME PAGE (Page 1)
# ===========================
def home_page():
    st.title(" Smart Mushroom Classifier")

    st.markdown("""
    <div style="text-align: center; margin-top: 50px; padding: 30px; border: 2px solid #e59866; border-radius: 15px; background: rgba(0, 0, 0, 0.7);">
        <h3 style="color: #f0f3f4;">Welcome to the Mushroom Classification Tool!</h3>
        <p style="color: #aeb6bf;">Use the mushroom's physical features to determine if it is Edible or Poisonous. The tool also explains why the prediction was made.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Classification", key="start_button"):
            st.session_state['page'] = 'classifier'
            st.rerun()


# ===========================
#     CLASSIFIER PAGE (Page 2)
# ===========================
def classifier_page():
    st.title(" Mushroom Feature Input")

    # Back button
    if st.button("⬅ Back to Home", key="back_button"):
        st.session_state['page'] = 'home'
        st.rerun()

    st.markdown("---")

    user_inputs = {}
    cols_count = 4
    cols = st.columns(cols_count)
    num_features = len(features)
    features_per_col = (num_features + cols_count - 1) // cols_count

    # Display inputs using st.selectbox
    for i in range(cols_count):
        with cols[i]:
            start_index = i * features_per_col
            end_index = min((i + 1) * features_per_col, num_features)

            for index in range(start_index, end_index):
                feature = features[index]
                options = MUSHROOM_MAPPING.get(feature, ["Unknown"])

                # Handle 'bruises' name correction if needed
                display_feature = feature.replace("-", " ").title()

                selected_option = st.selectbox(
                    label=display_feature,
                    options=options,
                    key=f"input_{feature}"
                )

                # Extract the character/symbol
                char_value = selected_option.split('(')[-1].replace(')', '').strip()
                if '?' in char_value:
                    char_value = '?'

                # Use 'b' for 'bruises' if input name was 'bruises' but stored in encoders as 'ruises'
                if feature == 'bruises' and 'ruises' in label_encoders:
                    # Use the correct key name from encoders if it's different
                    user_inputs['ruises'] = char_value
                else:
                    user_inputs[feature] = char_value

    st.markdown("---")

    if st.button("Predict & Explain", key="predict_button"):
        encoded = {}

        try:
            # 1. ENCODE INPUTS
            for col, le in label_encoders.items():
                if col == 'class': continue  # Skip target column

                # Determine the input value key (handle potential 'ruises'/'bruises' inconsistency)
                input_key = col
                if col == 'ruises':
                    input_key = 'bruises'

                input_value = user_inputs.get(input_key, 'p')  # Default to 'p' for veil-type if missing

                # Handle new unseen values (if the input value wasn't in the training data classes)
                # This ensures the transform method doesn't fail immediately on unseen characters
                if input_value not in le.classes_:
                    le.classes_ = np.append(le.classes_, input_value)

                encoded[col] = le.transform([input_value])[0]

            # 2. PREPARE DATAFRAME FOR MODEL
            input_df = pd.DataFrame([encoded])

            # Ensure column order matches the model's training data
            all_model_features_for_df = [c for c in ALL_MODEL_FEATURES if c != 'class']
            input_df = input_df[all_model_features_for_df]

            # =========================================================
            # 🛑 التعديل الرئيسي لحل مشكلة 'int' and 'str'
            # ---------------------------------------------------------
            for col in input_df.columns:
               
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0).astype(int)
            # =========================================================

            # 3. MAKE PREDICTION & CALCULATE CONFIDENCE
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0].max() * 100

            # 4. SHAP CALCULATION
            explainer = st.session_state.get('explainer')
            if explainer is None:
                st.session_state['page'] = 'explanation'
                st.session_state['prediction_result'] = prediction
                st.session_state['prediction_confidence'] = round(probability, 2)
                st.session_state['shap_percentages'] = None
                st.rerun()

            # --- Calculate SHAP Values ---
            with st.spinner("Calculating Feature Contributions (SHAP)..."):
                if st.session_state['explainer_type'] == 'tree':
                    # TreeExplainer expects features in correct order and type
                    shap_values = explainer.shap_values(input_df)
                    if isinstance(shap_values, list):
                        shap_sample_values = shap_values[prediction][0]
                    else:
                        shap_sample_values = shap_values[0]
                else:  # General Explainer
                    shap_values = explainer(input_df)
                    shap_sample_values = shap_values.values[0]

                # Convert SHAP Values to Percentages based on absolute contribution
                vals = np.abs(shap_sample_values)
                feature_names = input_df.columns

                # Check for zero sum case to prevent division by zero
                if vals.sum() == 0:
                    percentages = np.zeros_like(vals)
                else:
                    percentages = 100 * vals / vals.sum()

                # Create the final percentage DataFrame
                percentage_data = {
                    'Feature': feature_names,
                    'Contribution %': [round(p, 2) for p in percentages]
                }
                percentage_df = pd.DataFrame(percentage_data).sort_values(by='Contribution %', ascending=False)

            # 5. STORE RESULTS & NAVIGATE
            st.session_state['prediction_result'] = prediction
            st.session_state['prediction_confidence'] = round(probability, 2)
            st.session_state['shap_percentages'] = percentage_df

            st.session_state['page'] = 'explanation'
            st.rerun()

        except Exception as e:
            # Use st.exception for better error display in Streamlit GUI
            st.error("An unexpected error occurred during prediction/encoding. Details:")
            st.exception(e)


# ===========================
#     EXPLANATION PAGE (Page 3)
# ===========================
def explanation_page():
    # Retrieve results from Session State
    pred = st.session_state.get('prediction_result', 'N/A')
    confidence = st.session_state.get('prediction_confidence', 'N/A')
    percentage_df = st.session_state.get('shap_percentages', None)

    # Back button to return to input screen
    if st.button("⬅ Back to Input", key="back_from_expl"):
        st.session_state['page'] = 'classifier'
        st.rerun()

    st.markdown("---")

    # --- Result Header ---
    pred_label = 'Edible (e)' if pred == 0 else 'Poisonous (p)'

    if pred == 0:
        st.success(f"##  RESULT: {pred_label} (Confidence: {confidence}%)")
    else:
        st.error(f"##  RESULT: {pred_label} (Confidence: {confidence}%)")

    st.markdown("---")

    st.header(f"🔬 Why was the prediction {pred_label}?")
    st.markdown("The table below shows the features that contributed the most to this specific prediction.")

    if percentage_df is not None and not percentage_df.empty:

        # Filter out features with 0% contribution for clarity
        display_df = percentage_df[percentage_df['Contribution %'] > 0.0].head(10)

        st.dataframe(
            display_df,
            column_config={
                "Feature": "Feature",
                "Contribution %": st.column_config.ProgressColumn(
                    "Contribution %",
                    format="%.2f %%",
                    min_value=0,
                    # Max is calculated based on current display values for better visual
                    max_value=display_df['Contribution %'].max() if not display_df.empty else 100
                ),
            },
            hide_index=True,
            use_container_width=True
        )

        st.markdown(
            f"""
            <p style='font-size: 16px; color: #aeb6bf;'>
            *These percentages reflect the relative importance of each feature in driving the model's output
            towards the predicted class for this specific mushroom instance (based on absolute SHAP values).
            </p>
            """, unsafe_allow_html=True
        )
    else:
        st.warning(
            "Could not calculate feature contributions (SHAP values). Please ensure 'mushroom.csv' is in the directory.")


# ===========================
#     MAIN APP ROUTER
# ===========================
if st.session_state['page'] == 'home':
    home_page()
elif st.session_state['page'] == 'classifier':
    classifier_page()
elif st.session_state['page'] == 'explanation':
    explanation_page()