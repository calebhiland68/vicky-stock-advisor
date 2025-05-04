import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from gtts import gTTS
import os

# --- Fetch Stock Data ---
@st.cache_data
def get_stock_data(tickers, period="6mo"):
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, period=period, progress=False)
        if not df.empty:
            df['Ticker'] = ticker
            data[ticker] = df
    return data

# --- Prepare Features ---
def prepare_features(stock_data):
    features, labels, tickers = [], [], []
    for ticker, df in stock_data.items():
        df['Return'] = df['Close'].pct_change()
        df['Direction'] = (df['Return'].shift(-1) > 0).astype(int)
        df.dropna(inplace=True)
        X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = df['Direction']
        features.append(X)
        labels.append(y)
        tickers.extend([ticker] * len(X))
    return pd.concat(features), pd.concat(labels), tickers

# --- Train Model and Predict ---
def train_and_predict(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    return predictions, y_test, model

# --- Recommend Top Stocks ---
def recommend_top_stocks(tickers, predictions, threshold=0.6):
    df = pd.DataFrame({'Ticker': tickers, 'Prediction': predictions})
    df = df[df['Prediction'] > threshold]
    top_stocks = df.sort_values(by='Prediction', ascending=False).drop_duplicates('Ticker').head(3)
    return top_stocks

# --- Speak Response ---
def speak_text(text, filename="response.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    return filename

# --- Streamlit App ---
def main():
    st.title("Vicky AI Stock Advisor")
    st.write("Hello Mr. Broker, I'm Vicky. Let's find your top stock picks for today.")

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN', 'TSLA', 'META']
    data = get_stock_data(tickers)

    if st.button("Run Prediction"):
        X, y, ticker_list = prepare_features(data)
        predictions, y_test, model = train_and_predict(X, y)
        top_stocks = recommend_top_stocks(ticker_list, predictions)

        st.subheader("Top 3 Stock Picks:")
        for _, row in top_stocks.iterrows():
            st.write(f"- {row['Ticker']} with {row['Prediction']*100:.2f}% confidence")

        response = f"Top 3 picks are: {', '.join(top_stocks['Ticker'])}. Most confident is {top_stocks.iloc[0]['Ticker']}."
        voice_file = speak_text(response)
        st.audio(voice_file)

        selected = st.selectbox("Pick a stock to view recent performance:", tickers)
        if selected:
            chart_data = data[selected]
            st.line_chart(chart_data['Close'])
            st.dataframe(chart_data.tail(10))

    st.button("Refresh")

if __name__ == "__main__":
    main()
