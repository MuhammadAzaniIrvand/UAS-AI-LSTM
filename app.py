import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# STREAMLIT CONFIG - OPTIMIZE FOR FAST LOAD
# ==========================================
st.set_page_config(
    page_title="LSTM Sentiment Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide warnings
tf.get_logger().setLevel('ERROR')

st.title("🚗 Sentiment Prediksi Mobil Cina - LSTM Forecasting")
st.markdown("---")

# ==========================================
# LOAD MODEL & SCALER - WITH ERROR HANDLING
# ==========================================
@st.cache_resource
def load_model_and_scaler():
    """Load pre-trained model dan scaler - cached"""
    try:
        # Load model dengan TF lite optimization (faster)
        model = load_model('lstm_sentiment_model.h5', compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load metadata
        try:
            with open('model_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
        except:
            metadata = {'LOOK_BACK': 30, 'n_features': 4}
        
        return model, scaler, metadata, True, "✅ Model loaded successfully!"
    except FileNotFoundError as e:
        return None, None, None, False, f"❌ File tidak ditemukan: {str(e)}"
    except Exception as e:
        return None, None, None, False, f"❌ Error loading model: {str(e)}"

model, scaler, metadata, model_loaded, status_msg = load_model_and_scaler()

if not model_loaded:
    st.error(status_msg)
    st.error("""
    **File yang dibutuhkan:**
    - `lstm_sentiment_model.h5` 
    - `scaler.pkl` 
    - `model_metadata.pkl`
    - `data_with_sentiment_score.csv`
    """)
    st.stop()

st.success(status_msg)

LOOK_BACK = metadata.get('LOOK_BACK', 30)
n_features = metadata.get('n_features', 4)

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    """Load data CSV untuk inference"""
    try:
        df = pd.read_csv('data_with_sentiment_score.csv')
        df['published_at'] = pd.to_datetime(df['published_at']).dt.tz_localize(None)
        
        # Filter 2023+
        df_filtered = df[df['published_at'] >= '2023-01-01'].copy()
        df_daily = df_filtered.set_index('published_at').resample('D')['sentiment_score'].mean().ffill()
        df_smooth = df_daily.rolling(window=7).mean().fillna(method='bfill')
        
        return df_filtered, df_daily, df_smooth, True
    except FileNotFoundError:
        return None, None, None, False

df_filtered, df_daily, df_smooth, data_loaded = load_data()

if not data_loaded:
    st.error("❌ File 'data_with_sentiment_score.csv' tidak ditemukan!")
    st.stop()

# ==========================================
# SIDEBAR - USER INPUTS
# ==========================================
with st.sidebar:
    st.header("⚙️ Konfigurasi Prediksi")
    
    # Forecast horizon
    forecast_days = st.slider(
        "Prediksi berapa hari ke depan?",
        min_value=7,
        max_value=90,
        value=30,
        step=7,
        help="Semakin jauh prediksi, semakin besar uncertainty"
    )
    
    st.divider()
    
    # Display info
    st.subheader("📊 Data Info")
    st.metric("Total Days", len(df_smooth))
    st.metric("Current Sentiment", f"{df_smooth.values[-1]:.4f}")
    
    st.divider()
    
    # Model info
    st.subheader("🧠 Model Info")
    st.text("LSTM Architecture:")
    st.text("- LSTM(56) + Dropout(0.4)")
    st.text("- LSTM(28) + Dropout(0.4)")
    st.text("- Dense(14) + Dropout(0.25)")
    st.text("- Dense(1, sigmoid)")
    
    predict_button = st.button("🔮 Generate Forecast", use_container_width=True, type="primary")

# ==========================================
# DATA SUMMARY
# ==========================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Comments", f"{len(df_filtered):,}")
with col2:
    st.metric("Total Days", len(df_smooth))
with col3:
    st.metric("Avg Sentiment", f"{df_filtered['sentiment_score'].mean():.3f}")
with col4:
    st.metric("Sentiment Range", f"[{df_filtered['sentiment_score'].min():.3f}, {df_filtered['sentiment_score'].max():.3f}]")

st.markdown("---")

# ==========================================
# PREDICTION LOGIC
# ==========================================
if predict_button:
    with st.spinner("🔄 Preparing data..."):
        # Prepare features (4 features sesuai model)
        df_features = pd.DataFrame(index=df_smooth.index)
        df_features['smoothed'] = df_smooth
        df_features['raw_daily'] = df_daily
        df_features['momentum'] = df_smooth.diff().fillna(0)
        df_features['volatility'] = df_daily.rolling(window=7).std().fillna(0)
        
        scaled_data = scaler.transform(df_features)
        n_features = scaled_data.shape[1]
        LOOK_BACK = 30
        
        progress_bar = st.progress(30)
        st.info("✅ Data prepared")
    
    with st.spinner("🔮 Generating forecast..."):
        # Adaptive parameters based on horizon
        if forecast_days <= 30:
            mean_reversion = 0.05
            noise_level = 0.003
            momentum_damping = 0.30
            confidence_width = 0.025
            mode = "AGGRESSIVE (High Confidence)"
        elif forecast_days <= 60:
            mean_reversion = 0.08
            noise_level = 0.005
            momentum_damping = 0.20
            confidence_width = 0.045
            mode = "MODERATE"
        else:
            mean_reversion = 0.12
            noise_level = 0.007
            momentum_damping = 0.10
            confidence_width = 0.08
            mode = "CONSERVATIVE"
        
        # Statistics
        train_size = int(len(scaled_data) * 0.75)
        train_data_scaled = scaled_data[:train_size]
        global_mean = np.mean(train_data_scaled[:, 0])
        global_std = np.std(train_data_scaled[:, 0])
        
        # Get last window
        last_window = scaled_data[-LOOK_BACK:].copy()
        curr_input = last_window.reshape(1, LOOK_BACK, n_features)
        prev_val = curr_input[0, -1, 0]
        
        future_preds_scaled = []
        
        for day in range(forecast_days):
            # Predict
            pred_raw = model.predict(curr_input, verbose=0)[0, 0]
            
            # Mean reversion
            reversion_weight = mean_reversion * (1 + day * 0.02)
            reversion_weight = min(reversion_weight, 0.30)
            pred_revert = (pred_raw * (1 - reversion_weight)) + (global_mean * reversion_weight)
            
            # Add noise
            noise = np.random.normal(0, noise_level)
            pred_final = pred_revert + noise
            pred_final = np.clip(pred_final, global_mean - 3*global_std, global_mean + 3*global_std)
            
            future_preds_scaled.append(pred_final)
            
            # Update features
            feat_smoothed = pred_final
            feat_raw = pred_final + np.random.normal(0, noise_level * 2.5)
            feat_momentum = (pred_final - prev_val) * momentum_damping
            feat_volatility = global_std * np.random.uniform(0.8, 1.2)
            
            new_features = np.array([feat_smoothed, feat_raw, feat_momentum, feat_volatility])
            curr_input = np.append(curr_input[:, 1:, :], new_features.reshape(1, 1, n_features), axis=1)
            prev_val = pred_final
        
        # Inverse transform
        future_preds_dummy = np.zeros((len(future_preds_scaled), n_features))
        future_preds_dummy[:, 0] = future_preds_scaled
        future_preds = scaler.inverse_transform(future_preds_dummy)[:, 0]
        
        progress_bar.progress(100)
        st.success("✅ Forecast generated!")
    
    st.markdown("---")
    
    # ==========================================
    # RESULTS DISPLAY
    # ==========================================
    st.subheader(f"📈 Forecast Result ({mode})")
    
    # Mode info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Forecast Mode", mode.split('(')[0].strip())
    with col2:
        st.metric("Confidence Band", f"±{confidence_width:.3f}")
    with col3:
        st.metric("Forecast Days", forecast_days)
    
    st.markdown("---")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Full history + forecast
    ax1.plot(df_smooth.index, df_smooth.values, label='Historical Data', 
             color='blue', linewidth=2.5, alpha=0.8)
    
    future_dates = pd.date_range(start=df_smooth.index[-1] + timedelta(days=1), periods=forecast_days)
    ax1.plot(future_dates, future_preds, label=f'{forecast_days}-Day Forecast', 
             color='red', linewidth=2.5, linestyle='--', marker='o', markersize=4)
    
    global_mean_inv = scaler.inverse_transform(np.zeros((1, n_features)))[0, 0]
    # Adjust to get actual mean
    mean_dummy = np.zeros((1, n_features))
    mean_dummy[0, 0] = global_mean
    global_mean_inv = scaler.inverse_transform(mean_dummy)[0, 0]
    
    ax1.axhline(y=global_mean_inv, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Historical Mean')
    
    ax1.fill_between(future_dates, future_preds - confidence_width, future_preds + confidence_width, 
                     alpha=0.2, color='red', label='Confidence Band')
    
    ax1.set_title(f"LSTM Sentiment Forecast - {forecast_days} Days", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Sentiment Score", fontsize=11)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoom recent + forecast
    zoom_days = min(60, len(df_smooth))
    zoom_start = max(0, len(df_smooth) - zoom_days)
    
    ax2.plot(df_smooth.index[zoom_start:], df_smooth.values[zoom_start:], 
             label='Recent History', color='blue', linewidth=2.5, marker='.')
    ax2.plot(future_dates, future_preds, label='Forecast', 
             color='red', linewidth=2.5, linestyle='--', marker='o', markersize=4)
    ax2.axhline(y=global_mean_inv, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax2.fill_between(future_dates, future_preds - confidence_width, future_preds + confidence_width, 
                     alpha=0.2, color='red')
    
    ax2.set_title(f"Zoom: Last {zoom_days} Days + {forecast_days} Day Forecast", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Sentiment Score", fontsize=11)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Forecast summary
    st.subheader("📊 Forecast Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Sentiment", f"{df_smooth.values[-1]:.4f}")
    with col2:
        st.metric("Forecast Average", f"{np.mean(future_preds):.4f}")
    with col3:
        st.metric("Min Forecast", f"{np.min(future_preds):.4f}")
    with col4:
        st.metric("Max Forecast", f"{np.max(future_preds):.4f}")
    
    # Trend analysis
    first_10 = np.mean(future_preds[:min(10, len(future_preds))])
    last_10 = np.mean(future_preds[max(0, len(future_preds)-10):])
    trend = "📈 RISING" if last_10 > first_10 else "📉 DECLINING"
    change = ((last_10 - first_10) / abs(first_10) * 100) if first_10 != 0 else 0
    
    trend_col1, trend_col2 = st.columns(2)
    with trend_col1:
        st.metric("Trend (First vs Last 10 days)", trend)
    with trend_col2:
        st.metric("Change %", f"{change:+.2f}%")
    
    st.markdown("---")
    
    # Download forecast data
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': future_preds,
        'Lower_Band': future_preds - confidence_width,
        'Upper_Band': future_preds + confidence_width
    })
    
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast Data (CSV)",
        data=csv,
        file_name=f"sentiment_forecast_{forecast_days}days.csv",
        mime="text/csv"
    )
    
    # Show forecast table
    with st.expander("📋 View Forecast Table"):
        st.dataframe(forecast_df.reset_index(drop=True), use_container_width=True)

else:
    st.info("""
    👈 **Cara menggunakan:**
    1. Atur jumlah hari prediksi di sidebar (7-90 hari)
    2. Klik tombol "🔮 Generate Forecast"
    3. Lihat hasil visualisasi dan download data
    
    **Tips:**
    - 📈 **30 hari:** Confidence tinggi, gunakan untuk short-term planning
    - ⚠️ **60 hari:** Confidence sedang, good untuk trend analysis
    - ⚠️ **90 hari:** Confidence rendah, hanya untuk long-term reference
    """)
    