# 🚗 LSTM Sentiment Forecasting - Mobil Cina

Aplikasi prediksi sentiment mobil Cina menggunakan LSTM (Long Short-Term Memory) neural network.

## 📊 Features
- Prediksi sentiment berdasarkan data historis
- Visualisasi tren sentiment
- Forecasting untuk periode mendatang
- Interactive dashboard dengan Streamlit

## 🚀 Deploy ke Streamlit Cloud

### Cara Deploy:

1. **Fork/Push ke GitHub** ✅
   - Pastikan semua file sudah di-push ke GitHub

2. **Login ke Streamlit Cloud**
   - Kunjungi [share.streamlit.io](https://share.streamlit.io)
   - Login dengan akun GitHub

3. **Deploy App**
   - Klik "New app"
   - Pilih repository: `UAS-AI-LSTM`
   - Main file path: `app.py`
   - Klik "Deploy!"

4. **Tunggu 2-5 menit** ⏳
   - Streamlit akan install dependencies
   - App akan otomatis live!

## 📦 File Penting

- `app.py` - Main application
- `requirements.txt` - Dependencies
- `lstm_sentiment_model.h5` - Pre-trained LSTM model
- `scaler.pkl` - Data scaler
- `model_metadata.pkl` - Model metadata
- `data_with_sentiment_score.csv` - Dataset

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

## ⚠️ Catatan Deployment

- **File Size**: Model `.h5` harus < 100MB
- **RAM**: Free tier = 1GB (cukup untuk model ini)
- **Sleep Mode**: App sleep setelah tidak aktif (auto-wake saat diakses)

## 📝 Tech Stack

- **Framework**: Streamlit
- **ML Model**: TensorFlow/Keras LSTM
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib

---

Made with ❤️ for UAS AI
