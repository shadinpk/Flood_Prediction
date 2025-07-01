import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="🌧️ Flood Prediction System",
    page_icon="🌊",
    layout="wide"
)

# Error handling for model loading
@st.cache_resource
def load_prediction_model():
    try:
        model = load_model("flood_prediction_model.h5")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure 'flood_prediction_model.h5' and 'scaler.pkl' are in the directory.")
        return None, None

# Load model and scaler
model, scaler = load_prediction_model()

st.title("🌧️ Advanced Rainfall & Flood Prediction System")
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    simulation_variance = st.slider("Rainfall Simulation Variance", 0.1, 0.5, 0.2)
    flood_threshold = st.slider("Flood Probability Threshold", 0.3, 0.8, 0.5)
    show_advanced = st.checkbox("Show Advanced Metrics")

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📍 Location Input")
    place = st.text_input("Enter Place Name", placeholder="e.g., Mumbai, Kerala, Chennai")
    
    # Add location validation
    if place and len(place.strip()) < 2:
        st.warning("Please enter a valid place name (at least 2 characters)")

with col2:
    st.header("🌦️ Current Weather Data")
    manual_week = st.checkbox("Enter This Week's Rainfall Manually")
    
    if manual_week:
        week_rainfall = st.number_input(
            "Enter this week's rainfall (mm)", 
            min_value=0.0, 
            max_value=500.0,
            step=0.1,
            help="Enter observed rainfall for the current week"
        )
    else:
        # More realistic rainfall simulation based on season
        current_month = datetime.now().month
        if current_month in [6, 7, 8, 9]:  # Monsoon months
            base_rainfall = random.uniform(50, 200)
        elif current_month in [12, 1, 2]:  # Winter
            base_rainfall = random.uniform(5, 30)
        else:  # Other months
            base_rainfall = random.uniform(10, 80)
        
        week_rainfall = round(base_rainfall, 2)
        st.success(f"🌧️ Simulated rainfall for this week: **{week_rainfall} mm**")

# Monthly rainfall prediction
st.header("📊 12-Month Rainfall Forecast")

# More sophisticated monthly simulation
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# Seasonal patterns for more realistic simulation
seasonal_multipliers = {
    'JAN': 0.3, 'FEB': 0.4, 'MAR': 0.6, 'APR': 0.8,
    'MAY': 1.2, 'JUN': 2.5, 'JUL': 3.0, 'AUG': 2.8,
    'SEP': 2.2, 'OCT': 1.5, 'NOV': 0.8, 'DEC': 0.4
}

monthly_rainfall = []
for month in months:
    base_monthly = week_rainfall * 4 * seasonal_multipliers[month]
    varied_rainfall = base_monthly * random.uniform(1-simulation_variance, 1+simulation_variance)
    monthly_rainfall.append(round(max(0, varied_rainfall), 2))

rainfall_dict = {month: rain for month, rain in zip(months, monthly_rainfall)}
rainfall_df = pd.DataFrame([rainfall_dict])

# Enhanced visualizations
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📈 Monthly Rainfall Chart")
    fig = px.line(
        x=months, 
        y=monthly_rainfall,
        title="Predicted Monthly Rainfall",
        labels={'x': 'Month', 'y': 'Rainfall (mm)'}
    )
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Rainfall Distribution")
    fig = px.bar(
        x=months, 
        y=monthly_rainfall,
        title="Monthly Rainfall Distribution",
        labels={'x': 'Month', 'y': 'Rainfall (mm)'},
        color=monthly_rainfall,
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, use_container_width=True)

# Statistics
st.subheader("📋 Rainfall Statistics")
stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

with stats_col1:
    st.metric("Total Annual", f"{sum(monthly_rainfall):.1f} mm")
with stats_col2:
    st.metric("Average Monthly", f"{np.mean(monthly_rainfall):.1f} mm")
with stats_col3:
    st.metric("Peak Month", f"{months[np.argmax(monthly_rainfall)]}")
with stats_col4:
    st.metric("Peak Rainfall", f"{max(monthly_rainfall):.1f} mm")

# Monthly data table (collapsible)
with st.expander("📋 View Monthly Data Table"):
    st.dataframe(rainfall_df, use_container_width=True)

# Flood prediction
st.markdown("---")
st.header("🌊 Flood Risk Analysis")

if model is None or scaler is None:
    st.error("Cannot perform prediction: Model files not loaded")
elif st.button("🔍 Analyze Flood Risk", type="primary", use_container_width=True):
    if not place.strip():
        st.warning("⚠️ Please enter a place name to proceed with the analysis.")
    else:
        with st.spinner("Analyzing flood risk..."):
            # Prepare model input
            X_input = np.array(monthly_rainfall).reshape(1, -1)
            X_scaled = scaler.transform(X_input)

            # Flood prediction
            prediction_prob = model.predict(X_scaled)[0][0]
            is_flood_predicted = prediction_prob >= flood_threshold
            
            # Risk categorization
            if prediction_prob >= 0.8:
                risk_level = "🔴 VERY HIGH"
                risk_color = "red"
            elif prediction_prob >= 0.6:
                risk_level = "🟠 HIGH"
                risk_color = "orange"
            elif prediction_prob >= 0.4:
                risk_level = "🟡 MODERATE"
                risk_color = "gold"
            elif prediction_prob >= 0.2:
                risk_level = "🟢 LOW"
                risk_color = "green"
            else:
                risk_level = "🟢 VERY LOW"
                risk_color = "green"

            # Results display
            result_col1, result_col2 = st.columns([2, 1])
            
            with result_col1:
                st.subheader("🎯 Prediction Results")
                st.write(f"**📍 Location:** {place}")
                st.write(f"**🌧️ Current Week Rainfall:** {week_rainfall} mm")
                st.write(f"**🌊 Flood Prediction:** {'YES - Flood Expected!' if is_flood_predicted else 'NO - No Flood Expected'}")
                st.write(f"**📊 Risk Level:** {risk_level}")
                st.write(f"**🎯 Confidence Score:** {prediction_prob:.3f}")
                
                if show_advanced:
                    st.write(f"**🔧 Threshold Used:** {flood_threshold}")
                    st.write(f"**📈 Annual Rainfall:** {sum(monthly_rainfall):.1f} mm")
            
            with result_col2:
                # Risk gauge visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction_prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Flood Risk"},
                    gauge = {
                        'axis': {'range': [None, 1]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 0.2], 'color': "lightgreen"},
                            {'range': [0.2, 0.4], 'color': "yellow"},
                            {'range': [0.4, 0.6], 'color': "orange"},
                            {'range': [0.6, 1], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': flood_threshold
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            # Recommendations
            st.subheader("💡 Recommendations")
            if is_flood_predicted:
                st.error("""
                **Immediate Actions Recommended:**
                - 🚨 Monitor weather alerts and warnings
                - 📦 Prepare emergency supplies and evacuation kit
                - 🏠 Secure property and move valuables to higher ground
                - 📱 Stay connected with local authorities
                - 🚗 Plan alternative transportation routes
                """)
            else:
                st.success("""
                **Precautionary Measures:**
                - 📺 Continue monitoring weather conditions
                - 🔧 Ensure drainage systems are clear
                - 📋 Keep emergency contacts readily available
                - 🌧️ Stay informed about rainfall forecasts
                """)

            # Store history with timestamp
            if "predictions" not in st.session_state:
                st.session_state.predictions = []

            st.session_state.predictions.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Place": place,
                "Week Rainfall (mm)": week_rainfall,
                "Annual Rainfall (mm)": round(sum(monthly_rainfall), 1),
                "Flood Prediction": "YES" if is_flood_predicted else "NO",
                "Risk Level": risk_level,
                "Confidence": round(prediction_prob, 3),
                **rainfall_dict
            })

# Prediction history
st.markdown("---")
if st.checkbox("📚 Show Prediction History"):
    if st.session_state.get("predictions"):
        history_df = pd.DataFrame(st.session_state.predictions)
        st.subheader(f"📈 Prediction History ({len(history_df)} entries)")
        
        # Add filtering options
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            if st.button("🗑️ Clear History"):
                st.session_state.predictions = []
                st.rerun()
        
        with filter_col2:
            show_flood_only = st.checkbox("Show only flood predictions")
        
        if show_flood_only:
            history_df = history_df[history_df["Flood Prediction"] == "YES"]
        
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True)
            
            # Download option
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History as CSV",
                data=csv,
                file_name=f"flood_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No flood predictions found in history.")
    else:
        st.info("No predictions made yet. Make your first prediction above!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>
    🌧️ Advanced Rainfall & Flood Prediction System | 
    Built with Streamlit & TensorFlow | 
    For educational purposes only
    </small>
</div>
""", unsafe_allow_html=True)