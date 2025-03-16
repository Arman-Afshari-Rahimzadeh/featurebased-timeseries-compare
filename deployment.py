import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Deutsche Bank Net Income Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066b2;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #444;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .feature-section {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        position: relative;
    }
    .feature-section label {
        color: #444;
        font-weight: 500;
        margin-bottom: 5px;
        display: block;
    }
    .feature-section .stSlider {
        margin-top: 10px;
    }
    .stButton>button {
        background-color: #0066b2;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 5px;
        transition: all 0.3s;
        width: 100%;
        margin: 0 auto;
        display: block;
    }
    .stButton>button:hover {
        background-color: #004c87;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .prediction-text {
        text-align: center;
        margin: 20px auto;
        max-width: 800px;
        padding: 15px;
    }
    .spacer {
        height: 30px;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #666;
        font-size: 0.8rem;
    }
    .highlight {
        background-color: #e6f2ff;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    /* New styles for input symmetry */
    .input-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .input-label {
        font-weight: 500;
        margin-bottom: 8px;
        color: #333;
    }
    .stNumberInput > div > div > input {
        text-align: center;
    }
    div.stSlider > div {
        padding-left: 0;
        padding-right: 0;
    }
    div.stSlider > div > div > div > div {
        height: 100%;
    }
    /* Fix for slider alignment */
    div[data-testid="stSlider"] {
        padding-left: 0;
        padding-right: 0;
        margin-bottom: 25px;
    }
    /* Fix for number input alignment */
    div[data-testid="stNumberInput"] {
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('finalized_model.sav')
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to make predictions
def predict_net_income(features, model, scaler):
    try:
        # Convert input features to DataFrame
        input_df = pd.DataFrame([features])
        
        # Calculate missing features
        # For demonstration, we'll use placeholder values since we don't have historical data
        input_df['Net_Income_Rolling_Mean'] = 0  # This would normally be calculated from historical data
        input_df['Lag_Net_Income'] = 0  # This would normally be the previous day's net income
        input_df['Rolling_Operating_Income'] = input_df['Operating_Income']  # Simplified version
        
        # Add current month and year
        current_date = datetime.now()
        input_df['Month'] = current_date.month
        input_df['Year'] = current_date.year
        
        # Convert DataFrame to DMatrix
        dmatrix = xgb.DMatrix(input_df, enable_categorical=True)
        
        # Make prediction
        prediction = model.predict(dmatrix)
        pred_val = prediction[0]

        # Create an array with two columns, where the second value is a dummy (e.g., 0)
        dummy_input = np.array([[pred_val, 0]])

        # Perform the inverse transform on the dummy array
        descaled = scaler.inverse_transform(dummy_input)

        # Extract the descaled Net_Income (first column)
        descaled_net_income = descaled[0, 0]
        return descaled_net_income
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Function to create animated loading
def loading_animation():
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    return

# Main app
def main():
    # Header with animation
    st.markdown('<div class="main-header">Deutsche Bank Net Income Predictor</div>', unsafe_allow_html=True)
    
    
    # Load model
    model, scaler = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        return
    
    # Create sidebar for additional information
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/7b/Deutsche_Bank_logo_without_wordmark.svg", width=100)
        st.markdown("## About")
        st.info("This tool uses XGBoost to predict Deutsche Bank's Net Income based on historical financial data.")
        
        st.markdown("## Features Used")
        st.markdown("""
        - Operating Income
        - Expenses
        - Revenue
        - Interest Expense
        - Tax Expense
        - And more...
        """)
        
        st.markdown("## How to Use")
        st.markdown("""
        1. Adjust the sliders for each financial metric
        2. Click the 'Predict' button
        3. View the predicted Net Income and visualization
        """)
        
        st.markdown("## Model Information")
        st.markdown("**Model Type:** XGBoost Regressor")
        
    # Create a more balanced layout for the input fields
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # First column with input boxes
    with col1:
        st.markdown('<div class="sub-header">Financial Metrics (Part 1)</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            assets = st.number_input("Assets (in billions)", 
                            min_value=0.1, max_value=10.0, 
                            value=3.0, step=0.1) * 1_000_000_000
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            liabilities = st.number_input("Liabilities (in billions)", 
                                    min_value=0.025, max_value=10.0, 
                                    value=2.5, step=0.1) * 1_000_000_000
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            equity = st.number_input("Equity (in billions)", 
                            min_value=0.015, max_value=10.0, 
                            value=4.8, step=0.1) * 1_000_000_000
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            cash_flow = st.number_input("Cash Flow (in millions)", 
                                min_value=1.1, max_value=800.0, 
                                value=380.0, step=10.0) * 1_000_000
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Second column with input boxes
    with col2:
        st.markdown('<div class="sub-header">Financial Metrics (Part 2)</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            debt_to_equity = st.number_input("Debt to Equity Ratio", 
                                    min_value=1, max_value=3545, 
                                    value=343, step=10)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            interest_expense = st.number_input("Interest Expense (in millions)", 
                                        min_value=0.15, max_value=200.0, 
                                        value=90.0, step=5.0) * 1_000_000
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            tax_expense = st.number_input("Tax Expense (in millions)", 
                                    min_value=0.2, max_value=150.0, 
                                    value=67.0, step=5.0) * 1_000_000
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            dividend_payout = st.number_input("Dividend Payout (in millions)", 
                                min_value=1.2, max_value=300.0, 
                                value=133.0, step=5.0) * 1_000_000
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Third column with all sliders
    with col3:
        st.markdown('<div class="sub-header">Financial Metrics (Part 3)</div>', unsafe_allow_html=True)
        
        with st.container():
            operating_income = st.slider("Operating Income (in millions)", 
                                        min_value=2.9, max_value=1000.0, 
                                        value=500.0, step=10.0, 
                                        label_visibility="visible") * 1_000_000
            st.markdown('</div>', unsafe_allow_html=True)
            
            expenses = st.slider("Expenses (in millions)", 
                                min_value=0.8, max_value=500.0, 
                                value=250.0, step=10.0) * 1_000_000
            st.markdown('</div>', unsafe_allow_html=True)
            
            revenue = st.slider("Revenue (in millions)", 
                                min_value=2.4, max_value=1500.0, 
                                value=750.0, step=10.0) * 1_000_000
            st.markdown('</div>', unsafe_allow_html=True)
            
            roa = st.slider("Return on Assets (ROA)", 
                        min_value=-4, max_value=13, 
                        value=1, step=1)
            st.markdown('</div>', unsafe_allow_html=True)
            
            profit_margin = st.slider("Profit Margin", 
                                    min_value=-153, max_value=359, 
                                    value=27, step=5)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a spacer between input sections and button
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # Add a spacer before the button
    st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)
    
    # Center the predict button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("Predict Net Income", use_container_width=True)
    
    # Add a spacer after the button
    st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)
    
    # Create a dictionary with all features
    features = {
        'Operating_Income': operating_income,
        'Expenses': expenses,
        'Assets': assets,
        'Liabilities': liabilities,
        'Equity': equity,
        'Revenue': revenue,
        'Cash_Flow': cash_flow,
        'Debt_to_Equity': debt_to_equity,
        'ROA': roa,
        'Profit_Margin': profit_margin,
        'Interest_Expense': interest_expense,
        'Tax_Expense': tax_expense,
        'Dividend_Payout': dividend_payout
    }
    
    # Make prediction when button is clicked
    if predict_button:
        # Show loading animation
        with st.spinner("Calculating prediction..."):
            loading_animation()
            
            # Make prediction
            prediction = predict_net_income(features, model, scaler=scaler)
            
            if prediction is not None:
                # Display prediction in a nice box
                st.markdown(f"<h2 style='text-align: center; color: #0066b2;'>Predicted Net Income</h2>", unsafe_allow_html=True)
                
                # Format the prediction
                formatted_prediction = f"${prediction:,.2f}"
                prediction_in_millions = prediction / 1_000_000
                
                # Display the prediction with animation
                st.markdown(f"<h1 style='text-align: center; color: #004c87;'>{formatted_prediction}</h1>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Create visualizations
                st.markdown("### Visualization of Key Factors")
                
                # Create tabs for different visualizations
                tab1, tab2 = st.tabs(["Contribution Analysis", "Financial Overview"])
                
                with tab1:
                    # Create a bar chart showing the relative importance of each feature
                    # This is a simplified example - in a real app, you'd use feature importance from the model
                    feature_importance = {
                        'Operating Income': operating_income / 1_000_000,
                        'Expenses': -expenses / 1_000_000,  # Negative impact
                        'Revenue': revenue / 1_000_000,
                        'Interest Expense': -interest_expense / 1_000_000,  # Negative impact
                        'Tax Expense': -tax_expense / 1_000_000  # Negative impact
                    }
                    
                    fig = px.bar(
                        x=list(feature_importance.keys()),
                        y=list(feature_importance.values()),
                        title="Key Factors Affecting Net Income (in millions)",
                        labels={'x': 'Factor', 'y': 'Impact (in millions)'},
                        color=list(feature_importance.values()),
                        color_continuous_scale='RdBu',
                        template='plotly_white'
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Create a pie chart showing the breakdown of expenses
                    labels = ['Operating Costs', 'Interest Expense', 'Tax Expense', 'Other Expenses']
                    other_expenses = expenses - interest_expense - tax_expense
                    if other_expenses < 0:
                        other_expenses = expenses * 0.1  # Fallback if calculation is negative
                    
                    values = [
                        expenses - interest_expense - tax_expense - other_expenses,
                        interest_expense,
                        tax_expense,
                        other_expenses
                    ]
                    
                    # Convert to millions for display
                    values = [v / 1_000_000 for v in values]
                    
                    fig = px.pie(
                        values=values,
                        names=labels,
                        title="Expense Breakdown (in millions)",
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Blues_r
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add a gauge chart showing health of financial metrics
                st.markdown("### Financial Health Indicators")
                
                # Create metrics in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Debt-to-Equity Ratio", 
                        f"{debt_to_equity:.2f}", 
                        delta="Lower is better" if debt_to_equity < 500 else "High leverage",
                        delta_color="normal"
                    )
                
                with col2:
                    st.metric(
                        "Return on Assets", 
                        f"{roa:.2f}%", 
                        delta="↑" if roa > 1 else "↓",
                        delta_color="normal"
                    )
                
                with col3:
                    st.metric(
                        "Profit Margin", 
                        f"{profit_margin:.2f}%", 
                        delta="↑" if profit_margin > 20 else "↓",
                        delta_color="normal"
                    )
                
                # Add a section for historical comparison (simulated)
                st.markdown("### Historical Comparison")
                
                # Generate some simulated historical data
                dates = pd.date_range(end=datetime.now(), periods=10, freq='Q')
                historical_net_income = [
                    prediction * (1 + np.random.normal(-0.2, 0.2)) 
                    for _ in range(10)
                ]
                
                # Create a line chart
                hist_data = pd.DataFrame({
                    'Date': dates,
                    'Net Income': historical_net_income
                })
                
                # Add the current prediction
                current_date = pd.to_datetime(datetime.now())
                current_data = pd.DataFrame({
                    'Date': [current_date],
                    'Net Income': [prediction]
                })
                
                # Plot with Plotly
                fig = px.line(
                    hist_data, 
                    x='Date', 
                    y='Net Income',
                    title='Historical Net Income Trend with Current Prediction',
                    template='plotly_white'
                )
                
                # Add the current prediction as a marker
                fig.add_scatter(
                    x=current_data['Date'],
                    y=current_data['Net Income'],
                    mode='markers',
                    marker=dict(size=12, color='red'),
                    name='Current Prediction'
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a section for what-if analysis
                st.markdown("### What-If Analysis")
                st.markdown("Adjust key parameters to see how they affect Net Income")
                
                # Create columns for what-if sliders
                col1, col2 = st.columns(2)
                
                with col1:
                    revenue_change = st.slider(
                        "Revenue Change (%)", 
                        min_value=-50, 
                        max_value=50, 
                        value=0, 
                        step=5
                    )
                
                with col2:
                    expense_change = st.slider(
                        "Expense Change (%)", 
                        min_value=-50, 
                        max_value=50, 
                        value=0, 
                        step=5
                    )
                
                # Calculate new prediction with adjusted values
                if revenue_change != 0 or expense_change != 0:
                    # Create a copy of features
                    adjusted_features = features.copy()
                    
                    # Apply changes
                    adjusted_features['Revenue'] = features['Revenue'] * (1 + revenue_change/100)
                    adjusted_features['Expenses'] = features['Expenses'] * (1 + expense_change/100)
                    
                    # Make new prediction
                    adjusted_prediction = predict_net_income(adjusted_features, model)
                    
                    # Calculate the difference
                    difference = adjusted_prediction - prediction
                    percentage_change = (difference / prediction) * 100 if prediction != 0 else 0
                    
                    # Display the results
                    st.markdown(f"""
                    <div style='background-color: #f0f8ff; padding: 15px; border-radius: 5px;'>
                        <h4>What-If Analysis Results:</h4>
                        <p>Original Net Income: ${prediction:,.2f}</p>
                        <p>Adjusted Net Income: ${adjusted_prediction:,.2f}</p>
                        <p>Difference: ${difference:,.2f} ({percentage_change:.2f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()