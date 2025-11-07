# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from utils.inference import predict_fraud

# # Page config
# st.set_page_config(
#     page_title="Credit Card Fraud Detector",
#     page_icon="shield",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
#     <style>
#     .main-header {font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
#     .fraud {background-color: #ff6b6b; color: white; padding: 1rem; border-radius: 10px; text-align: center;}
#     .legit {background-color: #51cf66; color: white; padding: 1rem; border-radius: 10px; text-align: center;}
#     .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px;}
#     </style>
# """, unsafe_allow_html=True)

# # Header
# st.markdown('<h1 class="main-header">Credit Card Fraud Detector</h1>', unsafe_allow_html=True)
# st.markdown("**Powered by XGBoost** • Real-time fraud detection • Trained on 284K+ transactions")

# # Sidebar
# with st.sidebar:
#     st.header("How to Use")
#     st.markdown("""
#     1. **Pick a base transaction** from real test data
#     2. **Adjust Amount & Time** to simulate changes
#     3. **Click Predict** → Instant fraud score!
    
#     V1–V28 are auto-filled from real PCA features.
#     """)
#     st.markdown("---")
#     st.info("Capstone Project • 7th Sem Deep Learning")

# # Load demo data
# @st.cache_data
# def load_demo():
#     return pd.read_csv('data/demo_transactions.csv')

# demo_df = load_demo()

# # Main App
# col1, col2 = st.columns([1, 2])

# with col1:
#     st.header("Input Transaction")
    
#     # Select base transaction
#     selected_idx = st.selectbox(
#         "Choose base transaction:",
#         range(len(demo_df)),
#         # format_func=lambda i: f"Tx #{i+1} | €{demo_df.iloc[i]['Amount_scaled']:.2f} | V1={demo_df.iloc[i]['V1']:.2f}"
#         format_func=lambda i: f"Tx #{i+1} | €{demo_df.iloc[i]['Amount_scaled']:.2f} | {'FRAUD RISK' if i >= 10 else 'Normal'}"
#     )
    
#     base_row = demo_df.iloc[selected_idx]
    
#     # User inputs
#     amount = st.number_input("Transaction Amount (€)", value=89.50, min_value=0.0, step=0.01)
#     time_sec = st.number_input(
#     "Time (seconds since first transaction)", 
#     min_value=0, 
#     max_value=172792, 
#     value=50000, 
#     step=1
# )

    
#     # Extract V1-V28
#     v_features = {col: base_row[col] for col in demo_df.columns if col.startswith('V')}
    
#     if st.button("PREDICT FRAUD", type="primary", use_container_width=True):
#         with st.spinner("Analyzing..."):
#             result = predict_fraud(amount, time_sec, v_features)
        
#         # Result
#         st.markdown("### Prediction Result")
#         if result["probability"] > 0.5:
#             st.markdown(f'<div class="fraud"><h2>{result["label"]}</h2></div>', unsafe_allow_html=True)
#         else:
#             st.markdown(f'<div class="legit"><h2>{result["label"]}</h2></div>', unsafe_allow_html=True)
        
#         col_a, col_b, col_c = st.columns(3)
#         with col_a:
#             st.metric("Fraud Probability", f"{result['probability']:.1%}")
#         with col_b:
#             st.metric("Confidence", f"{result['confidence']}%")
#         with col_c:
#             st.metric("Risk Level", "HIGH" if result['probability'] > 0.5 else "LOW")

# with col2:
#     st.header("Fraud Confidence Gauge")
    
#     # Default gauge (before prediction)
#     if 'result' not in locals():
#         fig_gauge = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=0,
#             title={'text': "Awaiting Prediction"},
#             gauge={'axis': {'range': [0, 100]}}
#         ))
#     else:
#         fig_gauge = go.Figure(go.Indicator(
#             mode="gauge+number+delta",
#             value=result["confidence"],
#             domain={'x': [0, 1], 'y': [0, 1]},
#             title={'text': "Fraud Confidence"},
#             delta={'reference': 50},
#             gauge={
#                 'axis': {'range': [None, 100]},
#                 'bar': {'color': "darkblue"},
#                 'steps': [
#                     {'range': [0, 30], 'color': "lightgreen"},
#                     {'range': [30, 70], 'color': "yellow"},
#                     {'range': [70, 100], 'color': "red"}
#                 ],
#                 'threshold': {'line': {'color': "red", 'width': 4}, 'value': 70}
#             }
#         ))
    
#     st.plotly_chart(fig_gauge, use_container_width=True, config={'staticPlot': False})

#     # Feature Importance
#     st.markdown("### Top Predictive Features (XGBoost)")
#     if st.button("Show Feature Importance", use_container_width=True):
#         try:
#             st.image('outputs/plots/xgboost_feature_importance.png', use_column_width=True)
#         except:
#             st.warning("Feature importance plot not found. Run training script to generate.")

# # Footer
# st.markdown("---")
# st.caption("Dataset: ULB Credit Card Fraud • Model: XGBoost • Built with Streamlit")











#---------------------------------------------------------------------------------------------------------------------------------------









# WITH CSV 

# app.py
import streamlit as st
import pandas as pd
import joblib
from utils.preprocessor import prepare_input

# Load scaler and model
@st.cache_resource
def load_resources():
    scaler = joblib.load('utils/scaler.pkl')
    model = joblib.load('models/xgboost_model.pkl')
    return scaler, model

scaler, model = load_resources()

# Load original dataset
@st.cache_data
def load_original():
    return pd.read_csv('dataset/creditcard.csv')

df = load_original()

# Page config
st.set_page_config(page_title="Fraud Detector", page_icon="shield", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1f77b4;'>Credit Card Fraud Detector</h1>", unsafe_allow_html=True)
st.markdown("**Trained on 284K+ transactions • XGBoost • 89% Fraud Recall**")

# Tabs
tab1, tab2 = st.tabs(["Test from Original CSV", "Demo Transactions"])

# =====================================
# TAB 1: Test from Original CSV
# =====================================
# with tab1:
#     st.header("Test Any Row from `creditcard.csv`")
    
#     col1, col2 = st.columns([1, 2])
    
#     with col1:
#         row_idx = st.number_input(
#             "Enter Row Number (0 to 284806)",
#             min_value=0,
#             max_value=len(df)-1,
#             value=541,
#             step=1
#         )
        
#         if st.button("Load & Predict", type="primary", use_container_width=True):
#             row = df.iloc[row_idx]
            
#             # Extract features
#             amount = row['Amount']
#             time_sec = int(row['Time'])
#             v_features = {f'V{i}': row[f'V{i}'] for i in range(1, 29)}
            
#             # Predict
#             X = prepare_input(amount, time_sec, v_features)
#             proba = model.predict_proba(X)[0][1]
#             label = "FRAUD" if proba > 0.5 else "LEGITIMATE"
#             confidence = round(proba * 100, 2)
            
#             st.session_state.result = {
#                 "label": label,
#                 "confidence": confidence,
#                 "proba": proba,
#                 "amount": amount,
#                 "time": time_sec,
#                 "class": int(row['Class'])
#             }
    
#     with col2:
#         if 'result' in st.session_state:
#             res = st.session_state.result
#             color = "#ff6b6b" if res['label'] == "FRAUD" else "#51cf66"
#             st.markdown(f"<h2 style='color: {color}; text-align: center;'>{res['label']}</h2>", unsafe_allow_html=True)
            
#             col_a, col_b, col_c = st.columns(3)
#             with col_a:
#                 st.metric("Fraud Probability", f"{res['proba']:.1%}")
#             with col_b:
#                 st.metric("Confidence", f"{res['confidence']}%")
#             with col_c:
#                 st.metric("True Label", "FRAUD" if res['class'] == 1 else "LEGITIMATE")
            
#             st.info(f"**Amount**: €{res['amount']:.2f} | **Time**: {res['time']} sec")


with tab1:
    st.header("Test Any Row from `creditcard.csv`")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        total_rows = len(df)
    st.write(f"**Total Rows in CSV**: {total_rows} (including header)")

    row_idx = st.number_input(
        "Enter Row Number from CSV (1 = first transaction)",
        min_value=1,
        max_value=total_rows,
        value=1,  # Start with first transaction
        step=1,
        help="Row 1 → Time=0, Amount=149.62 | Row 2 → Time=1, Amount=378.66"
    )

    # CORRECT: Row 1 = iloc[0], Row 2 = iloc[1]
    pandas_idx = row_idx - 1
    row = df.iloc[pandas_idx]

    # Show preview
    st.write("**Preview of Selected Row:**")
    st.write(row[['Time', 'Amount', 'V1', 'V2', 'V3']].to_frame().T)

    if st.button("PREDICT FRAUD", type="primary", use_container_width=True):
        amount = row['Amount']
        time_sec = int(row['Time'])
        v_features = {f'V{i}': row[f'V{i}'] for i in range(1, 29)}

        X = prepare_input(amount, time_sec, v_features)
        proba = model.predict_proba(X)[0][1]
        label = "FRAUD" if proba > 0.5 else "LEGITIMATE"
        confidence = round(proba * 100, 2)

        st.session_state.result = {
            "label": label,
            "confidence": confidence,
            "proba": proba,
            "amount": amount,
            "time": time_sec,
            "class": int(row['Class']),
            "row": row_idx
        }
    
    with col2:
        if 'result' in st.session_state:
            res = st.session_state.result
            color = "#ff6b6b" if res['label'] == "FRAUD" else "#51cf66"
            st.markdown(f"<h2 style='color: {color}; text-align: center;'>🚨 {res['label']}</h2>", unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Fraud Probability", f"{res['proba']:.1%}")
            with col_b:
                st.metric("Confidence", f"{res['confidence']}%")
            with col_c:
                st.metric("True Label", "FRAUD" if res['class'] == 1 else "LEGITIMATE")
            
            st.info(f"**Row**: {res['row']} | **Amount**: €{res['amount']:.2f} | **Time**: {res['time']} sec")

# =====================================
# TAB 2: Demo Transactions
# =====================================
with tab2:
    st.header("Demo with 20 Real Transactions")
    
    demo_df = pd.read_csv('data/demo_transactions.csv')
    
    selected = st.selectbox(
        "Choose Transaction",
        range(len(demo_df)),
        format_func=lambda i: f"Tx #{i+1} | €{demo_df.iloc[i]['Amount_scaled']:.2f} | V1={demo_df.iloc[i]['V1']:.2f}"
    )
    
    row = demo_df.iloc[selected]
    amount = st.number_input("Amount (€)", value=float(row['Amount_scaled']), step=0.01)
    # time_sec = st.slider("Time (sec)", 0, 172792, value=50000)
    time_sec = st.number_input(
    "Time (seconds since first transaction)",
    min_value=0,
    max_value=172792,
    value=50000,
    step=1
)

    
    v_features = {col: row[col] for col in demo_df.columns if col.startswith('V')}
    
    if st.button("Predict", use_container_width=True):
        X = prepare_input(amount, time_sec, v_features)
        proba = model.predict_proba(X)[0][1]
        label = "FRAUD" if proba > 0.5 else "LEGITIMATE"
        st.markdown(f"### {label}")
        st.progress(float(proba))