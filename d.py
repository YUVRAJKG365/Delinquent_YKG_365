import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go


# ─── Load model ─────────────────────────────────────────────────────
try:
    model = joblib.load('bdelinquency_model.pkl')
    # Success indicator in sidebar
    st.sidebar.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model not found! Please train and upload `bdelinquency_model.pkl`.")
    st.stop()

# Custom styling
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Form styling */
    div[data-testid="stForm"] {
        border: 1px solid #e1e4e8;
        border-radius: 12px;
        padding: 25px;
        background: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Card styling for results */
    .custom-card {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* Custom progress bar */
    .stProgress > div > div > div {
        background-image: linear-gradient(to right, #4CAF50, #FFC107, #F44336);
    }
    
    /* Metric card styling */
    div[data-testid="metric-container"] {
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 15px;
        background: white !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Section header styling */
    .section-header {
        padding-bottom: 8px;
        border-bottom: 2px solid #0d6efd;
        margin-bottom: 20px;
        color: #0d6efd;
    }
</style>
""", unsafe_allow_html=True)

# ─── App header ─────────────────────────────────────────────────────
st.title("📊 Customer Delinquency Risk Assessment")
st.subheader("Predict the likelihood of customer delinquency using machine learning")

# Developer info and description
with st.expander("ℹ️ About this application", expanded=True):
    c1, c2 = st.columns([1, 3])
    with c1:
        st.image("https://img.icons8.com/fluency/96/user-male-circle.png", width=80)
    with c2:
        st.write("""
        **Developed by:** Yuvraj Gond  
        **Version:** 2.0 (Enhanced UI)  
        **Last updated:** July 2025
        """)
    
    st.write("""
    This application uses a predictive model to assess customer delinquency risk based on financial and behavioral factors.
    The model analyzes payment history, credit utilization, income, and other key indicators to provide risk assessments
    and actionable recommendations.
    """)

st.markdown("---")

# ─── Sidebar configuration ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    threshold = st.slider(
        "Delinquency Probability Threshold",
        min_value=0.0, max_value=1.0, value=0.50, step=0.01,
        help="Adjust the sensitivity for classifying customers as high-risk"
    )
    
    st.markdown("### 📚 How to Use")
    st.info("""
    1. Fill in customer details  
    2. Click **Predict Delinquency Risk**  
    3. Review risk level & recommendations  
    4. Adjust threshold if needed
    """)
    
    st.markdown("### ⚠️ Key Risk Indicators")
    st.caption("The model considers these high-risk factors:")
    st.markdown("- 🕒 3+ missed payments")
    st.markdown("- ⏱️ 2+ late payments")
    st.markdown("- 💳 Credit utilization > 70%")
    st.markdown("- 📉 Debt-to-income ratio > 40%")

# ─── Prediction form ────────────────────────────────────────────────
with st.form("prediction_form"):
    st.header("📝 Customer Information")
    
    # Personal details
    st.markdown('#### 👤 Personal Details')
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
    with c2:
        income = st.number_input(
            "Annual Income ($)", min_value=0.0, value=50000.0, format="%.2f"
        )
    with c3:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    
    # Financial metrics
    st.markdown('#### 💰 Financial Metrics')
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_util = st.number_input(
            "Credit Utilization (0–1)", min_value=0.0, max_value=1.0, value=0.30, format="%.2f"
        )
    with c2:
        loan_balance = st.number_input(
            "Loan Balance ($)", min_value=0.0, value=20000.0, format="%.2f"
        )
    with c3:
        debt_income = st.number_input(
            "Debt-to-Income (0–1)", min_value=0.0, max_value=1.0, value=0.30, format="%.2f"
        )
    
    # Account information
    st.markdown('#### 🏦 Account Information')
    c1, c2 = st.columns(2)
    with c1:
        tenure = st.number_input(
            "Account Tenure (months)", min_value=0, max_value=120, value=24
        )
    with c2:
        card = st.selectbox(
            "Credit Card Type",
            ["Standard", "Gold", "Platinum", "Student", "Business"]
        )
    
    # Payment history visualization
    st.markdown('#### 📅 Payment History (Last 6 Months)')
    months = [f"Month_{i}" for i in range(1,7)]
    statuses = {}
    cols = st.columns(6)
    
    # Color mapping for payment statuses
    status_colors = {
        "On-time": "#4CAF50",
        "Late": "#FFC107",
        "Missed": "#F44336"
    }
    
    for i, m in enumerate(months):
        with cols[i]:
            # Visual indicator for each month
            st.markdown(f"**Month {i+1}**")
            statuses[m] = st.selectbox(
                f"Select status for month {i+1}",
                ["On-time", "Late", "Missed"],
                key=m,
                label_visibility="collapsed"
            )
            # Color indicator
            color = status_colors[statuses[m]]
            st.markdown(f"<div style='background-color:{color}; height:5px; border-radius:2px;'></div>", 
                        unsafe_allow_html=True)
    
    # Additional information
    st.markdown('#### ℹ️ Additional Information')
    c1, c2 = st.columns(2)
    with c1:
        emp = st.selectbox(
            "Employment Status",
            ["Employed", "Self-employed", "Unemployed", "Retired"]
        )
    with c2:
        loc = st.selectbox(
            "Location",
            ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
        )
    
    # Submit button
    submitted = st.form_submit_button("🚀 Predict Delinquency Risk", use_container_width=True)

# ─── On submit: predict & display ───────────────────────────────────
if submitted:
    # Derive counts from dropdowns only
    late_count = sum(v == "Late" for v in statuses.values())
    missed_count = sum(v == "Missed" for v in statuses.values())
    ontime_count = sum(v == "On-time" for v in statuses.values())
    consistency = int(len(set(statuses.values())) == 1)

    # Build feature DataFrame
    X = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "Credit_Score": credit_score,
        "Credit_Utilization": credit_util,
        "Loan_Balance": loan_balance,
        "Debt_to_Income_Ratio": debt_income,
        "Account_Tenure": tenure,
        "Missed_Payments_Count": missed_count,
        "Late_Payments_Count": late_count,
        "OnTime_Payments_Count": ontime_count,
        "Payment_Consistency": consistency,
        "Employment_Status": emp,
        "Credit_Card_Type": card,
        "Location": loc
    }])

    # Predict probability & label
    prob = model.predict_proba(X)[0][1]
    pred_label = int(prob >= threshold)
    
    # Visual progress indicator
    st.markdown("#### 📊 Risk Probability")
    st.progress(prob, text=f"Delinquency Probability: {prob:.1%}")
    
    # Risk gauge visualization
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': "%"},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # Show result
    st.subheader("📋 Prediction Results")
    if pred_label:
        # High risk container
        st.markdown(f"""
        <div class="custom-card" style="background-color: #fff8f8; border-left: 4px solid #dc3545;">
            <h3 style="color: #dc3545;">🚨 High Risk of Delinquency ({prob:.0%} ≥ {threshold:.0%})</h3>
            <p><strong>Recommended Actions:</strong></p>
            <ul>
                <li>Proactive outreach & payment assistance</li>
                <li>Financial counseling enrollment</li>
                <li>Temporary payment relief programs</li>
                <li>Credit limit adjustment</li>
                <li>Enhanced account monitoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Low risk container
        st.markdown(f"""
        <div class="custom-card" style="background-color: #f8fff9; border-left: 4px solid #28a745;">
            <h3 style="color: #28a745;">✅ Low Risk of Delinquency ({(1-prob):.0%} < {threshold:.0%})</h3>
            <p><strong>Recommended Actions:</strong></p>
            <ul>
                <li>Standard account management</li>
                <li>Credit limit increase offers</li>
                <li>Cross-sell financial wellness tools</li>
                <li>Loyalty program enrollment</li>
                <li>Periodic account reviews</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Key metrics
    st.subheader("📈 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Missed Payments", f"{missed_count}/6", "High risk" if missed_count >= 3 else "Normal")
    col2.metric("Credit Utilization", f"{credit_util:.0%}", "High" if credit_util > 0.7 else "Normal")
    col3.metric("Debt-to-Income", f"{debt_income:.0%}", "High" if debt_income > 0.4 else "Normal")
    # Display payment history
    st.subheader("📅 Payment History Overview")
    payment_history = pd.DataFrame({
        "Month": months,
        "Status": list(statuses.values())
    })
    
    # Risk factors analysis
    st.subheader("🔍 Risk Factor Analysis")
    factors = []
    risk_score = 0
    
    if missed_count >= 3:
        factors.append(f"• **{missed_count} missed payments** (High risk)")
        risk_score += 1
    elif missed_count > 0:
        factors.append(f"• **{missed_count} missed payments** (Moderate risk)")
    
    if late_count >= 2:
        factors.append(f"• **{late_count} late payments** (High risk)")
        risk_score += 1
    elif late_count > 0:
        factors.append(f"• **{late_count} late payments** (Moderate risk)")
    
    if credit_util > 0.7:
        factors.append(f"• **High credit utilization** ({credit_util:.0%})")
        risk_score += 1
    elif credit_util > 0.5:
        factors.append(f"• **Moderate credit utilization** ({credit_util:.0%})")
    
    if debt_income > 0.4:
        factors.append(f"• **High debt-to-income ratio** ({debt_income:.0%})")
        risk_score += 1
    elif debt_income > 0.3:
        factors.append(f"• **Moderate debt-to-income ratio** ({debt_income:.0%})")
    
    if not factors:
        factors.append("• No significant risk factors identified")
    
    # Display risk factors
    with st.expander(f"View detailed analysis (Risk score: {risk_score}/4)", expanded=True):
        st.markdown("\n".join(factors))
        
        # Risk score visualization
        if risk_score > 0:
            risk_levels = ["Low", "Moderate", "Elevated", "High", "Critical"]
            st.progress(risk_score/4, text=f"Overall Risk Level: {risk_levels[min(risk_score, 4)]}")

# ─── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.caption("© 2025 Financial Analytics Department | Predictive Collections Optimization | v2.0")