try:
    from importlib import import_module

    st = import_module("streamlit")
except ModuleNotFoundError as exc:
    if exc.name == "streamlit":
        raise RuntimeError(
            "Streamlit is not installed. Install the app dependencies with "
            "`python -m pip install streamlit pandas numpy plotly joblib`, "
            "then run `python -m streamlit run app.py`."
        ) from exc
    raise
try:
    pd = import_module("pandas")
except ModuleNotFoundError as exc:
    if exc.name == "pandas":
        raise RuntimeError(
            "Pandas is not installed. Install the app dependencies with "
            "`python -m pip install pandas numpy plotly joblib`."
        ) from exc
    raise
try:
    np = import_module("numpy")
except ModuleNotFoundError as exc:
    if exc.name == "numpy":
        raise RuntimeError(
            "NumPy is not installed. Install the app dependencies with "
            "`python -m pip install numpy pandas plotly joblib`."
        ) from exc
    raise
try:
    go = import_module("plotly.graph_objects")
except ModuleNotFoundError as exc:
    if exc.name == "plotly":
        raise RuntimeError(
            "Plotly is not installed. Install the app dependencies with "
            "`python -m pip install plotly`."
        ) from exc
    raise
from datetime import datetime

# --- 1. PAGE CONFIGURATION & TERMINAL STYLING ---
st.set_page_config(page_title="Credit Risk Terminal", layout="wide")

# This CSS block forces a dark, terminal-like aesthetic
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { color: #00FF00 !important; font-family: 'Courier New', monospace; }
</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR: MODEL INPUTS ---
st.sidebar.title("Model Inputs (USD)")
st.sidebar.markdown("Adjust these sliders to stress-test the company.")

ticker = st.sidebar.text_input("Ticker Symbol", value="DAL")
market_cap = st.sidebar.number_input("Market Cap (MM)", value=17992.0, step=100.0)
share_price = st.sidebar.number_input("Share Price", value=28.06, step=1.0)
price_vol = st.sidebar.slider("Price Volatility (1-Yr) %", min_value=0.0, max_value=100.0, value=48.19)
long_term_debt = st.sidebar.number_input("Long-Term Debt (MM)", value=35728.0, step=100.0)
short_term_debt = st.sidebar.number_input("Short-Term Debt (MM)", value=2662.0, step=100.0)

# Auto-calculate total debt
total_debt = long_term_debt + short_term_debt
st.sidebar.markdown(f"**Total Debt:** `{total_debt:,.2f} MM`")

# --- 3. MAIN DASHBOARD AREA ---
st.title(f"Terminal: {ticker} Credit Risk Scorecard")
st.markdown("---")

# --- 4. TOP METRICS (THE SCORECARD) ---
col1, col2, col3, col4 = st.columns(4)

# A simulated math formula to calculate default probability based on your inputs
simulated_default_prob = (total_debt / market_cap) * (price_vol / 100) * 1.5

with col1:
    st.metric(label="1-Yr Default Prob", value=f"{simulated_default_prob:.2f}%", delta="-0.15%")
with col2:
    st.metric(label="5-Yr Market CDS", value="542 bps", delta="12 bps")
with col3:
    st.metric(label="Debt/Equity (%)", value=f"{(total_debt/market_cap)*100:.1f}", delta="-5.2")
with col4:
    st.metric(label="Share Price", value=f"${share_price}", delta="0.54")

st.markdown("---")

# --- 5. INTERACTIVE TERMINAL CHART ---
st.subheader("1-Yr Default Prob vs Share Price History")

# Generate fake historical data so the chart looks alive
dates = pd.date_range(end=datetime.today(), periods=200)
mock_price = np.linspace(40, share_price, 200) + np.random.normal(0, 1.5, 200)
mock_risk = np.linspace(1.0, simulated_default_prob, 200) + np.random.normal(0, 0.05, 200)

# Build a dual-axis chart with Plotly
fig = go.Figure()

# Add Share Price Line (White)
fig.add_trace(go.Scatter(
    x=dates, y=mock_price, name="Share Price", line=dict(color="white", width=2)
))

# Add Default Risk Line (Green) on a secondary Y-axis
fig.add_trace(go.Scatter(
    x=dates, y=mock_risk, name="Default Prob (%)", line=dict(color="#00FF00", width=1), yaxis="y2"
))

# Style the chart to match the terminal look
fig.update_layout(
    plot_bgcolor="black", paper_bgcolor="black", font=dict(color="white"),
    margin=dict(l=20, r=20, t=30, b=20),
    yaxis=dict(title="Share Price ($)", gridcolor="#333333"),
    yaxis2=dict(title="Default Prob (%)", overlaying="y", side="right", gridcolor="#333333"),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)")
)

# Display the chart on the Streamlit app
st.plotly_chart(fig, use_container_width=True)

# --- 6. SECTOR COMPARISON TABLE ---
st.subheader("Sector Comparison | Transportation & Logistics")

# Creating a mock dataset to exactly match the bottom left of your screenshot
sector_data = {
    "Credit Metric": ["Debt/Equity (%)", "EBIT/Int Exp", "Int Coverage", "ROA (%)"],
    "DAL (Selected)": [854.7, 5.6, 5.6, 0.8],
    "10th Percentile": [8.0, -1.5, -1.4, -3.9],
    "90th Percentile": [589.4, 107.3, 107.3, 20.7]
}
df_sector = pd.DataFrame(sector_data)

# Displaying it cleanly in Streamlit without the row numbers (index)
st.dataframe(df_sector, use_container_width=True, hide_index=True)

# --- 7. MACHINE LEARNING MODEL INTEGRATION ---
st.markdown("---")
st.subheader("🧠 Live AI Model Prediction")
st.markdown("This connects directly to the Random Forest model you built in `train.py`.")

try:
    import joblib
    import os
    
    # Load the model you trained in step 1
    model_path = "models/default_model.pkl"
    model = joblib.load(model_path)
    
    # The model expects 5 specific columns: 
    # ['debt_to_income', 'revolving_utilization', 'age', 'monthly_income', 'open_credit_lines']
    # We map your slider inputs into a format the model understands:
    live_features = pd.DataFrame({
        'debt_to_income': [(total_debt / market_cap) * 100], # Derived from sliders
        'revolving_utilization': [price_vol / 100],          # Derived from sliders
        'age': [45],                                         # Placeholder
        'monthly_income': [8500],                            # Placeholder
        'open_credit_lines': [7]                             # Placeholder
    })
    
    # Ask the ML model for a real prediction
    ml_prediction = model.predict_proba(live_features)[0][1]
    
    # Display the result
    if ml_prediction > 0.5:
        st.error(f"High Risk Detected! The AI predicts a **{ml_prediction * 100:.2f}%** chance of default.")
    else:
        st.success(f"Safe! The AI predicts a **{ml_prediction * 100:.2f}%** chance of default.")
        
except FileNotFoundError:
    st.warning("⚠️ ML Model not found. Did you run `python train.py` first to generate the .pkl file?")


    # --- 8. MONTE CARLO CREDIT RISK SIMULATION ---
st.markdown("---")
st.subheader("🎲 Monte Carlo Default Risk Simulation")

mc_paths = st.sidebar.slider("Monte Carlo Simulation Paths", min_value=100, max_value=2000, value=500, step=100)
trading_days = 252
dt = 1 / trading_days
mu = 0.05  # 5% drift assumption

# Generate random normal distribution shocks
np.random.seed(42)
shocks = np.random.normal(0, 1, (trading_days, mc_paths))

# Calculate daily price paths via GBM
sigma = price_vol / 100
price_matrix = np.zeros((trading_days, mc_paths))
price_matrix[0] = share_price

for t in range(1, trading_days):
    drift = (mu - 0.5 * (sigma ** 2)) * dt
    diffusion = sigma * np.sqrt(dt) * shocks[t]
    price_matrix[t] = price_matrix[t - 1] * np.exp(drift + diffusion)

# Define default barrier based on leverage ratio
leverage_ratio = total_debt / (market_cap + total_debt)
default_barrier = share_price * leverage_ratio

# Identify paths that breached the threshold
defaulted_paths = np.any(price_matrix <= default_barrier, axis=0)
mc_default_rate = (np.sum(defaulted_paths) / mc_paths) * 100

st.metric("Simulated 1-Yr Breach Probability", f"{mc_default_rate:.2f}%", help="Percentage of paths crossing below debt barrier")

# Render simulation plot (first 100 paths for performance)
fig_mc = go.Figure()
display_paths = min(100, mc_paths)

for i in range(display_paths):
    line_color = "red" if defaulted_paths[i] else "rgba(0, 255, 0, 0.15)"
    fig_mc.add_trace(go.Scatter(
        y=price_matrix[:, i], mode="lines",
        line=dict(width=1, color=line_color),
        showlegend=False
    ))

# Add Default Barrier Line
fig_mc.add_hline(
    y=default_barrier, line_dash="dash", line_color="red",
    annotation_text=f"Default Threshold (${default_barrier:.2f})"
)

fig_mc.update_layout(
    plot_bgcolor="black", paper_bgcolor="black", font=dict(color="white"),
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(title="Trading Days", gridcolor="#222222"),
    yaxis=dict(title="Share Price ($)", gridcolor="#222222")
)

st.plotly_chart(fig_mc, use_container_width=True)