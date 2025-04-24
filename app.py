# app.py
# Note: Remember to modify the api_key as described in the system documentation
# Authors: Du Hongzhou & Qi Legan (Team: Studio 6324)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import datetime
import joblib
import os
import tensorflow as tf
import time
from alpha_vantage.timeseries import TimeSeries
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import shap
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import ta

# Set Seaborn theme
sns.set_style('darkgrid')  # or use sns.set_theme()
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


# Function: Download stock data (Alpha Vantage)
@st.cache_data(ttl=3600)  # Cache data for one hour
def download_stock_data_alpha_vantage(ticker, start_date, end_date, api_key, retries=3, delay=5):
    ts = TimeSeries(key=api_key, output_format='pandas')
    for attempt in range(retries):
        try:
            # Get daily stock data
            data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
            # Rename columns to match yfinance format
            data = data.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            # Convert index to datetime
            data.index = pd.to_datetime(data.index)
            # Filter date range
            data = data[(data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))]
            if data.empty:
                st.error(
                    f"Could not download data for {ticker}. Please check if the ticker symbol is correct or adjust the date range.")
            return data
        except Exception as e:
            st.warning(f"Error downloading data: {e}, attempting to download again ({attempt + 1}/{retries})...")
            time.sleep(delay)
    st.error(
        "Could not download data after multiple attempts. Please try again later or check your network connection.")
    return pd.DataFrame()


# Function: Get macroeconomic data (example)
@st.cache_data
def get_macro_data():
    if os.path.exists('macro_data.csv'):
        macro_data = pd.read_csv('macro_data.csv', parse_dates=['Date'])
        return macro_data
    else:
        st.warning("Macroeconomic data file 'macro_data.csv' does not exist.")
        return None


# Function: Check time series stationarity
def check_stationarity(time_series, window=12):
    # Perform Augmented Dickey-Fuller test
    result = adfuller(time_series.dropna())

    # Create result dictionary
    adf_result = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Lags': result[2],
        'Observations': result[3],
        'Critical Value (1%)': result[4]['1%'],
        'Critical Value (5%)': result[4]['5%'],
        'Critical Value (10%)': result[4]['10%']
    }

    # Perform seasonal decomposition
    try:
        decomposition = seasonal_decompose(time_series.dropna(), model='additive', period=window)
        return adf_result, decomposition
    except:
        return adf_result, None


# Function: Calculate Commodity Channel Index (CCI)
def calculate_CCI(data, window=20):
    TP = (data['High'] + data['Low'] + data['Close']) / 3
    rolling_mean = TP.rolling(window=window).mean()
    rolling_std = TP.rolling(window=window).std()
    CCI = (TP - rolling_mean) / (0.015 * rolling_std)
    return CCI


# Function: Calculate Average Directional Index (ADX)
def calculate_ADX(data, window=14):
    plus_dm = data['High'].diff()
    minus_dm = data['Low'].diff().abs() * -1

    # Calculate directional movement indicators
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm < plus_dm) & (minus_dm < 0), 0)

    # Calculate True Range (TR)
    tr1 = data['High'] - data['Low']
    tr2 = (data['High'] - data['Close'].shift(1)).abs()
    tr3 = (data['Low'] - data['Close'].shift(1)).abs()
    TR = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate smoothed indicators
    ATR = TR.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / ATR)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / ATR)

    # Calculate ADX
    DX = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).abs()
    ADX = DX.rolling(window=window).mean()

    return ADX, plus_di, minus_di


# Function: Calculate Average True Range (ATR)
def calculate_ATR(data, window=14):
    high_low = data['High'] - data['Low']
    high_close_prev = np.abs(data['High'] - data['Close'].shift())
    low_close_prev = np.abs(data['Low'] - data['Close'].shift())
    TR = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    ATR = TR.rolling(window=window, min_periods=window).mean()
    return ATR


# Function: Calculate VWAP (Volume Weighted Average Price)
def calculate_VWAP(data):
    data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (data['TP'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    return data['VWAP']


# Function: Calculate Accumulation/Distribution Line
def calculate_ADL(data):
    mfm = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    mfm = mfm.fillna(0.0)  # Handle potential division by zero
    mfv = mfm * data['Volume']
    adl = mfv.cumsum()
    return adl


# Function: Calculate Relative Strength Index (multiple periods)
def calculate_RSI(data, periods=[14, 7, 21]):
    rsi_data = {}
    close_delta = data['Close'].diff()

    for period in periods:
        # Calculate ups and downs
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)

        # Calculate EMA
        ma_up = up.ewm(com=period - 1, adjust=True, min_periods=period).mean()
        ma_down = down.ewm(com=period - 1, adjust=True, min_periods=period).mean()

        # Calculate RSI
        rsi = 100 - (100 / (1 + ma_up / ma_down))
        rsi_data[f'RSI_{period}'] = rsi

    return pd.DataFrame(rsi_data)


# Function: Data preprocessing
def preprocess_data(data, macro_data=None):
    # Check and fill missing values
    data = data.copy()
    data.fillna(method='ffill', inplace=True)

    # Ensure 'Close' is a one-dimensional Series
    if isinstance(data['Close'].values, np.ndarray) and data['Close'].values.ndim > 1:
        data['Close'] = pd.Series(data['Close'].values.squeeze(), index=data.index)

    # Add debugging information
    st.write(f"data['Close'] type: {type(data['Close'])}")
    st.write(f"data['Close'] shape: {data['Close'].shape}")

    # Calculate technical indicators
    # Use Ta-Lib complete indicator library + custom indicators

    # 1. Trend indicators
    # RSI (multiple periods)
    rsi_df = calculate_RSI(data, periods=[7, 14, 21])
    for col in rsi_df.columns:
        data[col] = rsi_df[col]

    # MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Diff'] = data['MACD'] - data['MACD_Signal']

    # Moving Averages (multiple periods)
    windows = [5, 10, 20, 50, 100, 200]
    for window in windows:
        data[f'MA{window}'] = data['Close'].rolling(window=window).mean()
        data[f'EMA{window}'] = data['Close'].ewm(span=window, adjust=False).mean()

    # 2. Volatility indicators
    # Bollinger Bands
    for window in [20, 10, 50]:
        data[f'Bollinger_Middle_{window}'] = data['Close'].rolling(window=window).mean()
        data[f'Bollinger_Std_{window}'] = data['Close'].rolling(window=window).std()
        data[f'Bollinger_High_{window}'] = data[f'Bollinger_Middle_{window}'] + (2 * data[f'Bollinger_Std_{window}'])
        data[f'Bollinger_Low_{window}'] = data[f'Bollinger_Middle_{window}'] - (2 * data[f'Bollinger_Std_{window}'])
        data[f'Bollinger_Width_{window}'] = data[f'Bollinger_High_{window}'] - data[f'Bollinger_Low_{window}']
        data[f'Bollinger_%B_{window}'] = (data['Close'] - data[f'Bollinger_Low_{window}']) / (
                    data[f'Bollinger_High_{window}'] - data[f'Bollinger_Low_{window}'])

    # 3. Momentum indicators
    # Stochastic Oscillator
    for window in [14, 7, 21]:
        low_min = data['Low'].rolling(window=window).min()
        high_max = data['High'].rolling(window=window).max()
        data[f'Stochastic_%K_{window}'] = ((data['Close'] - low_min) / (high_max - low_min)) * 100
        data[f'Stochastic_%D_{window}'] = data[f'Stochastic_%K_{window}'].rolling(window=3).mean()

    # 4. Volume indicators
    # On-Balance Volume (OBV)
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv

    # Chaikin Money Flow (CMF)
    for window in [20, 10, 30]:
        mfv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (
                    data['High'] - data['Low'] + 1e-12)  # Avoid division by zero
        mfv = mfv.fillna(0)
        data[f'Chaikin_MF_{window}'] = (mfv * data['Volume']).rolling(window=window).sum() / data['Volume'].rolling(
            window=window).sum()

    # Accumulation/Distribution Line
    data['ADL'] = calculate_ADL(data)

    # VWAP
    data['VWAP'] = calculate_VWAP(data)

    # 5. Volatility indicators
    # ATR
    data['ATR'] = calculate_ATR(data)

    # 6. Trend direction indicators
    # ADX
    data['ADX'], data['Plus_DI'], data['Minus_DI'] = calculate_ADX(data)

    # 7. Other indicators
    # Ichimoku Cloud
    tenkan_window = 9
    kijun_window = 26
    senkou_span_b_window = 52

    tenkan_sen = (data['High'].rolling(window=tenkan_window).max() + data['Low'].rolling(
        window=tenkan_window).min()) / 2
    kijun_sen = (data['High'].rolling(window=kijun_window).max() + data['Low'].rolling(window=kijun_window).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_window)
    senkou_span_b = ((data['High'].rolling(window=senkou_span_b_window).max() + data['Low'].rolling(
        window=senkou_span_b_window).min()) / 2).shift(kijun_window)

    data['Ichimoku_Tenkan'] = tenkan_sen
    data['Ichimoku_Kijun'] = kijun_sen
    data['Ichimoku_A'] = senkou_span_a
    data['Ichimoku_B'] = senkou_span_b
    data['Ichimoku_Chikou'] = data['Close'].shift(-kijun_window)  # Lagging span

    # CCI
    data['CCI'] = calculate_CCI(data)

    # Rate of Change (ROC) - multiple periods
    for window in [1, 5, 10, 20, 60]:
        data[f'ROC_{window}'] = data['Close'].pct_change(window) * 100

    # Volatility
    for window in [5, 10, 20, 30, 60]:
        data[f'Volatility_{window}'] = data['Close'].rolling(window=window).std() / data['Close'].rolling(
            window=window).mean() * 100

    # Calculate trend change rate
    for window in [5, 10, 20]:
        data[f'Trend_Change_{window}'] = data['Close'].diff(window) / data['Close'].shift(window) * 100

    # Calculate daily returns
    data['Daily_Return'] = data['Close'].pct_change()

    # Calculate log returns
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))

    # Calculate cumulative returns
    data['Cum_Return'] = (1 + data['Daily_Return']).cumprod()

    # Position relative to N-day high/low
    for window in [10, 20, 50, 100]:
        data[f'Price_High_Ratio_{window}'] = data['Close'] / data['High'].rolling(window=window).max()
        data[f'Price_Low_Ratio_{window}'] = data['Close'] / data['Low'].rolling(window=window).min()

    # Volume relative change
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Volume_MA10'] = data['Volume'].rolling(window=10).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA10']

    # Short-term mean reversion indicator
    data['Mean_Reversion_3'] = (data['Close'] - data['Close'].rolling(window=3).mean()) / data['Close'].rolling(
        window=3).std()

    # Price pattern recognition features
    data['Doji'] = ((data['Close'] - data['Open']).abs() / (data['High'] - data['Low'])) < 0.1
    data['Long_Body'] = ((data['Close'] - data['Open']).abs() / (data['High'] - data['Low'])) > 0.7

    # Remove missing values
    data.dropna(inplace=True)

    # If macroeconomic data is available, merge it
    if macro_data is not None:
        data = data.merge(macro_data, on='Date', how='left')
        data.fillna(method='ffill', inplace=True)
        data.dropna(inplace=True)

    # Confirm 'ATR' was successfully added
    if 'ATR' not in data.columns:
        st.error("ATR calculation failed, 'ATR' column does not exist. Please check the data preprocessing steps.")
    else:
        st.write("'ATR' calculated successfully, data columns include:")
        st.write(data.columns.tolist())

    return data


# Function: Feature selection
def select_features(X, y, n_features=20):
    # 1. F-test based feature selection
    f_selector = SelectKBest(f_regression, k=n_features)
    f_selector.fit(X, y)
    f_scores = pd.DataFrame({'Feature': X.columns, 'F_Score': f_selector.scores_})
    f_scores = f_scores.sort_values('F_Score', ascending=False)

    # 2. Mutual information based feature selection
    mi_selector = SelectKBest(mutual_info_regression, k=n_features)
    mi_selector.fit(X, y)
    mi_scores = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_selector.scores_})
    mi_scores = mi_scores.sort_values('MI_Score', ascending=False)

    # 3. Random Forest feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = pd.DataFrame({'Feature': X.columns, 'RF_Importance': rf.feature_importances_})
    rf_importance = rf_importance.sort_values('RF_Importance', ascending=False)

    # 4. XGBoost feature importance
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X, y)
    xgb_importance = pd.DataFrame({'Feature': X.columns, 'XGB_Importance': xgb_model.feature_importances_})
    xgb_importance = xgb_importance.sort_values('XGB_Importance', ascending=False)

    # Combined ranking
    combined_scores = f_scores.merge(mi_scores, on='Feature') \
        .merge(rf_importance, on='Feature') \
        .merge(xgb_importance, on='Feature')

    # Normalize scores
    for col in ['F_Score', 'MI_Score', 'RF_Importance', 'XGB_Importance']:
        combined_scores[f'{col}_Norm'] = (combined_scores[col] - combined_scores[col].min()) / (
                    combined_scores[col].max() - combined_scores[col].min())

    # Calculate total score
    combined_scores['Total_Score'] = combined_scores['F_Score_Norm'] + combined_scores['MI_Score_Norm'] + \
                                     combined_scores['RF_Importance_Norm'] + combined_scores['XGB_Importance_Norm']

    # Sort by total score
    combined_scores = combined_scores.sort_values('Total_Score', ascending=False)

    # Select top n_features
    selected_features = combined_scores.head(n_features)['Feature'].tolist()

    return selected_features, combined_scores


# Function: Create LSTM dataset
def create_lstm_dataset(data, time_step=60, features=None, target='Close'):
    """
    Create time series dataset for LSTM
    """
    if features is None:
        features = list(data.columns)
        if target in features:
            features.remove(target)

    # Add target variable
    features = [target] + features

    # Extract feature data
    dataset = data[features].values

    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i + time_step])
        y.append(dataset[i + time_step, 0])  # Target variable is the first column

    return np.array(X), np.array(y)


# Function: Create dataset for traditional machine learning models
def create_ml_dataset(data, features, target='Close', lag_periods=[1, 5, 10, 20]):
    """
    Create dataset for traditional machine learning models, including lagged features
    """
    X = data[features].copy()
    y = data[target]

    # Add lagged features
    for period in lag_periods:
        for feature in features:
            if feature != target:  # Avoid lagging the target variable
                X[f'{feature}_lag_{period}'] = data[feature].shift(period)

    # Remove missing values
    X = X.dropna()
    y = y.loc[X.index]

    return X, y


# Function: Build and train advanced LSTM model
def build_and_train_advanced_lstm(X_train, y_train, X_val, y_val, epochs=50, batch_size=64, patience=10):
    """
    Build and train advanced LSTM model, including bidirectional LSTM and attention mechanism
    """
    # Define early stopping and learning rate reduction callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience // 2, min_lr=0.0001)

    # Create model
    model = Sequential()

    # First layer: Bidirectional LSTM
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))

    # Second layer: Regular LSTM
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))

    # Fully connected layers
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))  # Output layer

    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    return model, history


# Function: Build and train CNN-LSTM hybrid model
def build_and_train_cnn_lstm(X_train, y_train, X_val, y_val, epochs=50, batch_size=64, patience=10):
    """
    Build and train CNN-LSTM hybrid model for capturing local and global patterns in time series
    """
    # Define early stopping and learning rate reduction callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience // 2, min_lr=0.0001)

    # Create model
    model = Sequential()

    # Convolutional layers for extracting local features
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
                     input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # LSTM layers for capturing sequential features
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.3))

    # Fully connected layers
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))  # Output layer

    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    return model, history


# Function: Build and train GRU model
def build_and_train_gru(X_train, y_train, X_val, y_val, epochs=50, batch_size=64, patience=10):
    """
    Build and train GRU model as an alternative to LSTM
    """
    # Define early stopping and learning rate reduction callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience // 2, min_lr=0.0001)

    # Create model
    model = Sequential()

    # GRU layers
    model.add(GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(GRU(64, return_sequences=False))
    model.add(Dropout(0.3))

    # Fully connected layers
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))  # Output layer

    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    return model, history


# Function: Build and train enhanced traditional machine learning models
def build_and_train_advanced_ml_models(X_train, y_train, X_val=None, y_val=None, cv=5):
    """
    Build and train enhanced traditional machine learning models, including hyperparameter tuning and model ensembling
    """
    # Base models
    base_models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Support Vector Machine': SVR(),
        'XGBoost': xgb.XGBRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42)
    }

    # Hyperparameter grids
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'Support Vector Machine': {
            'kernel': ['rbf', 'poly'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'colsample_bytree': [0.7, 0.8, 0.9]
        },
        'AdaBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0]
        }
    }

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv)

    # Train and tune models
    trained_models = {}
    best_params = {}
    cv_scores = {}

    for name, model in base_models.items():
        # Use grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        # Save best model and parameters
        best_model = grid_search.best_estimator_
        trained_models[name] = best_model
        best_params[name] = grid_search.best_params_

        # Calculate cross-validation scores
        cv_score = cross_val_score(best_model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
        cv_scores[name] = -cv_score.mean()  # Convert back to MSE

    # Create voting regressor
    estimators = [(name, model) for name, model in trained_models.items()]
    voting_regressor = VotingRegressor(estimators=estimators)
    voting_regressor.fit(X_train, y_train)

    # Add voting regressor to model dictionary
    trained_models['Voting Ensemble'] = voting_regressor

    # If validation set is provided, calculate validation scores
    val_scores = {}
    if X_val is not None and y_val is not None:
        for name, model in trained_models.items():
            predictions = model.predict(X_val)
            val_scores[name] = mean_squared_error(y_val, predictions)

    return trained_models, best_params, cv_scores, val_scores


# Function: Enhanced risk assessment
def assess_risk_advanced(real, predictions, window=20, threshold_volatility=1.5, threshold_drawdown=0.05,
                         confidence_level=0.95):
    """
    Enhanced risk assessment, considering volatility, drawdown, and VaR
    """
    # Create result DataFrame
    df = pd.DataFrame({'Real': real, 'Predicted': predictions})

    # Calculate prediction error
    df['Error'] = df['Predicted'] - df['Real']
    df['Percent_Error'] = df['Error'] / df['Real'] * 100

    # Calculate volatility
    df['Volatility'] = df['Real'].rolling(window=window).std() / df['Real'].rolling(window=window).mean()

    # Calculate historical max versus current value (drawdown)
    df['Cummax'] = df['Real'].cummax()
    df['Drawdown'] = (df['Cummax'] - df['Real']) / df['Cummax']

    # Calculate daily returns
    df['Daily_Return'] = df['Real'].pct_change()

    # Calculate historical VaR
    df['VaR'] = df['Daily_Return'].rolling(window=window).quantile(1 - confidence_level)

    # Risk assessment
    # 1. Error-based risk
    df['Risk_Error'] = np.where(np.abs(df['Percent_Error']) > threshold_volatility * df['Volatility'] * 100, 'High',
                                'Medium')

    # 2. Volatility-based risk
    df['Risk_Volatility'] = np.where(df['Volatility'] > threshold_volatility, 'High', 'Medium')

    # 3. Drawdown-based risk
    df['Risk_Drawdown'] = np.where(df['Drawdown'] > threshold_drawdown, 'High', 'Medium')

    # 4. VaR-based risk
    df['Risk_VaR'] = np.where(df['Daily_Return'] < df['VaR'], 'High', 'Medium')

    # Comprehensive risk rating
    risk_columns = ['Risk_Error', 'Risk_Volatility', 'Risk_Drawdown', 'Risk_VaR']
    df['Risk_Count'] = df[risk_columns].apply(lambda x: (x == 'High').sum(), axis=1)

    df['Risk_Level'] = pd.cut(
        df['Risk_Count'],
        bins=[-1, 0, 1, 2, 4],
        labels=['Low', 'Medium', 'High', 'Very High']
    )

    return df


# Function: Calculate VaR and CVaR at different confidence levels
def calculate_var_cvar(returns, confidence_levels=[0.95, 0.99]):
    """
    Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR) at different confidence levels
    """
    var_results = {}
    cvar_results = {}

    for level in confidence_levels:
        # Calculate VaR
        var = returns.quantile(1 - level)
        var_results[f'VaR_{int(level * 100)}'] = var

        # Calculate CVaR (also known as Expected Shortfall)
        cvar = returns[returns <= var].mean()
        cvar_results[f'CVaR_{int(level * 100)}'] = cvar

    return var_results, cvar_results


# Function: Monte Carlo simulation
def monte_carlo_simulation(data, n_simulations=1000, n_days=30, confidence_level=0.95):
    """
    Use Monte Carlo simulation to forecast future stock price movements and risk
    """
    # Calculate log returns
    returns = np.log(1 + data['Close'].pct_change()).dropna()

    # Calculate returns mean and standard deviation
    mu = returns.mean()
    sigma = returns.std()

    # Last closing price
    last_price = data['Close'].iloc[-1]

    # Simulate paths
    simulation_results = []

    for _ in range(n_simulations):
        # Generate random normal returns
        random_returns = np.random.normal(mu, sigma, n_days)

        # Calculate price path
        price_path = [last_price]

        for ret in random_returns:
            price_path.append(price_path[-1] * np.exp(ret))

        simulation_results.append(price_path)

    # Convert to DataFrame
    sim_df = pd.DataFrame(simulation_results).T

    # Calculate daily VaR
    daily_var = {}
    for i in range(1, n_days + 1):
        daily_prices = sim_df.loc[i]
        daily_returns = (daily_prices - last_price) / last_price
        var = np.percentile(daily_returns, (1 - confidence_level) * 100)
        daily_var[i] = var

    return sim_df, daily_var


# Function: Stress testing
def stress_test(model, data, features, target='Close', scenarios=None):
    """
    Perform stress testing to simulate model performance under extreme market conditions
    """
    if scenarios is None:
        # Default stress scenarios
        scenarios = {
            'Mild Bear Market': {'factor': 0.95, 'volatility': 1.2},  # Price down 5%, volatility up 20%
            'Severe Bear Market': {'factor': 0.8, 'volatility': 1.5},  # Price down 20%, volatility up 50%
            'Market Crash': {'factor': 0.5, 'volatility': 2.0},  # Price down 50%, volatility doubled
            'Mild Bull Market': {'factor': 1.05, 'volatility': 0.9},  # Price up 5%, volatility down 10%
            'Strong Bull Market': {'factor': 1.2, 'volatility': 0.8}  # Price up 20%, volatility down 20%
        }

    # Prepare test data
    test_data = data.copy()

    results = {}

    for scenario_name, params in scenarios.items():
        # Apply stress scenario
        scenario_data = test_data.copy()

        # Price-related features multiplied by factor
        price_features = ['Open', 'High', 'Low', 'Close']
        for feature in price_features:
            if feature in scenario_data.columns:
                scenario_data[feature] *= params['factor']

        # Volatility-related features multiplied by volatility factor
        volatility_features = [col for col in scenario_data.columns if
                               'volatility' in col.lower() or 'std' in col.lower()]
        for feature in volatility_features:
            scenario_data[feature] *= params['volatility']

        # Prepare prediction data
        X = scenario_data[features]
        y_true = scenario_data[target]

        # Prediction
        y_pred = model.predict(X)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Save results
        results[scenario_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Predictions': y_pred
        }

    return results


# Function: Plot predictions (Plotly)
def plot_predictions(real, predictions, ticker, confidence_interval=None):
    """
    Use Plotly to plot prediction results, optionally showing confidence intervals
    """
    fig = go.Figure()

    # Plot real values
    fig.add_trace(go.Scatter(
        x=real.index,
        y=real,
        mode='lines',
        name='Actual Close Price',
        line=dict(color='blue', width=2)
    ))

    # Plot predicted values
    fig.add_trace(go.Scatter(
        x=predictions.index,
        y=predictions,
        mode='lines',
        name='Predicted Close Price',
        line=dict(color='red', width=2)
    ))

    # If confidence interval is provided, add confidence bands
    if confidence_interval is not None:
        lower_bound = confidence_interval['lower']
        upper_bound = confidence_interval['upper']

        fig.add_trace(go.Scatter(
            x=predictions.index,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=predictions.index,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            name='95% Confidence Interval'
        ))

    # Update layout
    fig.update_layout(
        title=f'{ticker} Close Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(x=0, y=1),
        template='plotly_white',
        hovermode='x unified'
    )

    # Add range selector
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    st.plotly_chart(fig)


# Function: Plot risk heatmap
def plot_risk_heatmap(risk_df, ticker):
    """
    Visualize risk distribution using a heatmap
    """
    # Prepare heatmap data
    heatmap_data = pd.crosstab(
        risk_df['Risk_Volatility'],
        risk_df['Risk_Error'],
        values=risk_df['Drawdown'],
        aggfunc='mean'
    )

    # Create heatmap
    fig = px.imshow(
        heatmap_data,
        title=f'{ticker} Risk Heatmap (color represents average drawdown)',
        color_continuous_scale='RdYlGn_r',  # Red to green color map, red indicates high risk
        labels=dict(x="Prediction Error Risk", y="Volatility Risk", color="Avg Drawdown")
    )

    # Update layout
    fig.update_layout(
        xaxis_title='Prediction Error Risk',
        yaxis_title='Volatility Risk',
        template='plotly_white'
    )

    st.plotly_chart(fig)


# Function: Plot Monte Carlo simulation results
def plot_monte_carlo(sim_df, ticker, last_price, confidence_level=0.95):
    """
    Plot Monte Carlo simulation results
    """
    fig = go.Figure()

    # Add all simulation paths (to reduce clutter, only show 100 paths)
    num_paths_to_show = min(100, sim_df.shape[1])

    for i in range(num_paths_to_show):
        fig.add_trace(go.Scatter(
            y=sim_df[i],
            mode='lines',
            line=dict(width=0.5, color='rgba(70, 130, 180, 0.2)'),
            showlegend=False
        ))

    # Add mean path
    mean_path = sim_df.mean(axis=1)
    fig.add_trace(go.Scatter(
        y=mean_path,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Mean Forecast Path'
    ))

    # Add confidence interval
    upper = sim_df.quantile(confidence_level, axis=1)
    lower = sim_df.quantile(1 - confidence_level, axis=1)

    fig.add_trace(go.Scatter(
        y=upper,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        y=lower,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0, 100, 80, 0.2)',
        name=f'{int(confidence_level * 100)}% Confidence Interval'
    ))

    # Add starting point
    fig.add_trace(go.Scatter(
        x=[0],
        y=[last_price],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Current Price'
    ))

    # Update layout
    days = list(range(sim_df.shape[0]))
    fig.update_layout(
        title=f'{ticker} Monte Carlo Simulation ({sim_df.shape[1]} simulations)',
        xaxis_title='Days',
        yaxis_title='Forecasted Price',
        legend=dict(x=0, y=1),
        template='plotly_white'
    )

    fig.update_xaxes(tickvals=days[::5])

    st.plotly_chart(fig)


# Function: Plot stress test results
def plot_stress_test_results(stress_results, ticker):
    """
    Visualize stress test results
    """
    # Extract evaluation metrics
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    scenarios = list(stress_results.keys())

    results_dict = {metric: [] for metric in metrics}

    for scenario in scenarios:
        for metric in metrics:
            results_dict[metric].append(stress_results[scenario][metric])

    # Create facet bar chart
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=scenarios,
            y=results_dict[metric],
            name=metric,
            marker_color=colors[i % len(colors)]
        ))

    # Update layout
    fig.update_layout(
        title=f'{ticker} Stress Test Results',
        xaxis_title='Scenario',
        yaxis_title='Metric Value',
        barmode='group',
        template='plotly_white'
    )

    st.plotly_chart(fig)


# Function: Plot feature importance visualization
def plot_feature_importance(importance_df, title="Feature Importance"):
    """
    Plot feature importance bar chart, including results from multiple methods
    """
    # Select top 20 features (if more than 20)
    if len(importance_df) > 20:
        importance_df = importance_df.head(20)

    # Create subplots
    fig = go.Figure()

    # Add a bar for each importance metric
    if 'F_Score' in importance_df.columns:
        fig.add_trace(go.Bar(
            y=importance_df['Feature'],
            x=importance_df['F_Score'],
            name='F-test Score',
            orientation='h',
            marker=dict(color='rgba(58, 71, 80, 0.6)')
        ))

    if 'MI_Score' in importance_df.columns:
        fig.add_trace(go.Bar(
            y=importance_df['Feature'],
            x=importance_df['MI_Score'],
            name='Mutual Information Score',
            orientation='h',
            marker=dict(color='rgba(246, 78, 139, 0.6)')
        ))

    if 'RF_Importance' in importance_df.columns:
        fig.add_trace(go.Bar(
            y=importance_df['Feature'],
            x=importance_df['RF_Importance'],
            name='Random Forest Importance',
            orientation='h',
            marker=dict(color='rgba(6, 147, 227, 0.6)')
        ))

    if 'XGB_Importance' in importance_df.columns:
        fig.add_trace(go.Bar(
            y=importance_df['Feature'],
            x=importance_df['XGB_Importance'],
            name='XGBoost Importance',
            orientation='h',
            marker=dict(color='rgba(153, 221, 255, 0.6)')
        ))

    if 'Total_Score' in importance_df.columns:
        fig.add_trace(go.Bar(
            y=importance_df['Feature'],
            x=importance_df['Total_Score'],
            name='Combined Score',
            orientation='h',
            marker=dict(color='rgba(255, 102, 0, 0.6)')
        ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        barmode='group',
        height=600,
        template='plotly_white'
    )

    st.plotly_chart(fig)


# Function: Plot SHAP values
def plot_shap_values(model, X, plot_type='summary'):
    """
    Calculate and plot SHAP values for model interpretation
    """
    # Calculate SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Create plot
    if plot_type == 'summary':
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type="bar")
        st.pyplot(plt)
    elif plot_type == 'dependence':
        # Select most important feature
        feature_names = X.columns
        mean_shap = np.abs(shap_values.values).mean(0)
        most_important = feature_names[np.argmax(mean_shap)]

        plt.figure(figsize=(10, 8))
        shap.dependence_plot(most_important, shap_values.values, X, feature_names=feature_names)
        st.pyplot(plt)
    elif plot_type == 'waterfall':
        # Create waterfall plot for first instance
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_values[0])
        st.pyplot(plt)
    else:
        st.error(f"Unknown SHAP plot type: {plot_type}")


# Function: Plot advanced technical indicators
def plot_advanced_technical_indicators(data, ticker):
    """
    Use Plotly to plot advanced technical indicators
    """
    # Create main candlestick chart
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick',
        increasing_line=dict(color='green'),
        decreasing_line=dict(color='red')
    ))

    # Add moving averages
    ma_periods = [20, 50, 100, 200]
    colors = ['blue', 'orange', 'purple', 'brown']

    for i, period in enumerate(ma_periods):
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[f'MA{period}'],
            mode='lines',
            name=f'MA{period}',
            line=dict(color=colors[i % len(colors)], width=1)
        ))

    # Add Bollinger Bands
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Bollinger_High_20'],
        mode='lines',
        line=dict(color='rgba(173,216,230,0.7)', width=1),
        name='Bollinger Upper'
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Bollinger_Low_20'],
        mode='lines',
        line=dict(color='rgba(173,216,230,0.7)', width=1),
        fill='tonexty',
        name='Bollinger Lower'
    ))

    # Add Ichimoku Cloud
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Ichimoku_A'],
        mode='lines',
        line=dict(color='rgba(255,99,71,0.5)', width=1),
        name='Senkou Span A'
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Ichimoku_B'],
        mode='lines',
        line=dict(color='rgba(60,179,113,0.5)', width=1),
        fill='tonexty',
        name='Senkou Span B'
    ))

    # Update layout
    fig.update_layout(
        title=f'{ticker} Advanced Technical Indicators',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        legend=dict(x=0, y=1),
        hovermode='x unified'
    )

    # Add range selector
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    st.plotly_chart(fig)

    # Create multi-indicator subplots
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('RSI Indicator', 'MACD Indicator', 'ADX Indicator', 'Cumulative Returns')
    )

    # RSI chart
    for period in [7, 14, 21]:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[f'RSI_{period}'],
                mode='lines',
                name=f'RSI_{period}'
            ),
            row=1, col=1
        )

    # Add RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)

    # MACD chart
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue')
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MACD_Signal'],
            mode='lines',
            name='MACD Signal',
            line=dict(color='red')
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['MACD_Diff'],
            name='MACD Histogram',
            marker=dict(
                color=np.where(data['MACD_Diff'] >= 0, 'green', 'red')
            )
        ),
        row=2, col=1
    )

    # ADX chart
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['ADX'],
            mode='lines',
            name='ADX',
            line=dict(color='purple')
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Plus_DI'],
            mode='lines',
            name='+DI',
            line=dict(color='green')
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Minus_DI'],
            mode='lines',
            name='-DI',
            line=dict(color='red')
        ),
        row=3, col=1
    )

    # Add ADX reference line
    fig.add_hline(y=25, line_dash="dash", line_color="grey", row=3, col=1)

    # Cumulative returns chart
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Cum_Return'],
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='blue')
        ),
        row=4, col=1
    )

    # Update layout
    fig.update_layout(
        title=f'{ticker} Technical Indicator Details',
        height=1200,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig)

    # Create other indicator charts
    # ATR chart
    fig_atr = px.line(
        data,
        x=data.index,
        y='ATR',
        title='ATR (Average True Range)',
        template='plotly_white'
    )

    # Add range selector
    fig_atr.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    st.plotly_chart(fig_atr)

    # CCI chart
    fig_cci = px.line(
        data,
        x=data.index,
        y='CCI',
        title='CCI (Commodity Channel Index)',
        template='plotly_white'
    )

    # Add CCI reference lines
    fig_cci.add_hline(y=100, line_dash="dash", line_color="red")
    fig_cci.add_hline(y=-100, line_dash="dash", line_color="green")

    st.plotly_chart(fig_cci)

    # VWAP chart
    fig_vwap = go.Figure()

    fig_vwap.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue')
    ))

    fig_vwap.add_trace(go.Scatter(
        x=data.index,
        y=data['VWAP'],
        mode='lines',
        name='VWAP',
        line=dict(color='orange')
    ))

    fig_vwap.update_layout(
        title='Close Price vs VWAP (Volume Weighted Average Price)',
        template='plotly_white'
    )

    st.plotly_chart(fig_vwap)


# Main function
def main():
    st.title('Financial Risk Assessment and Early Warning System - Team: Studio 6324')
    st.markdown("""
    This system analyzes stock market data using multiple machine learning models to predict future closing prices and assess risk levels.
    It integrates a multi-dimensional technical indicator framework, supports deep learning and ensemble learning models, and provides various risk quantification methods and visualization tools.
    """)

    st.sidebar.header('Parameters')

    # User inputs
    ticker = st.sidebar.text_input('Stock Symbol', 'AAPL').upper()
    start_date = st.sidebar.date_input('Start Date', datetime.date(2015, 1, 1))
    end_date = st.sidebar.date_input('End Date', datetime.date.today())

    # Create sidebar tabs
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Basic Analysis", "Advanced Analysis", "Risk Assessment", "Monte Carlo Simulation", "Feature Importance",
         "Model Interpretation"]
    )

    # Common parameters
    time_step = st.sidebar.slider('Time Step (Days)', min_value=30, max_value=120, value=60, step=10)

    # Display different parameters based on analysis mode
    if analysis_mode == "Basic Analysis":
        pass  # Use default parameters

    elif analysis_mode == "Advanced Analysis":
        epochs = st.sidebar.slider('Deep Learning Training Epochs', min_value=20, max_value=200, value=50, step=10)
        batch_size = st.sidebar.slider('Batch Size', min_value=16, max_value=128, value=64, step=16)
        patience = st.sidebar.slider('Early Stopping Patience', min_value=5, max_value=30, value=10, step=5)
        model_type = st.sidebar.selectbox(
            'Deep Learning Model Type',
            ["LSTM", "Bidirectional LSTM", "GRU", "CNN-LSTM"]
        )
        feature_selection = st.sidebar.checkbox('Enable Automatic Feature Selection', True)
        n_features = st.sidebar.slider('Number of Features', min_value=10, max_value=50, value=20, step=5)

    elif analysis_mode == "Risk Assessment":
        risk_threshold = st.sidebar.slider('Risk Threshold (Return)', min_value=0.01, max_value=0.10, value=0.02,
                                           step=0.01)
        confidence_level = st.sidebar.slider('VaR Confidence Level', min_value=0.90, max_value=0.99, value=0.95,
                                             step=0.01)
        window = st.sidebar.slider('Risk Assessment Window', min_value=10, max_value=60, value=20, step=5)
        threshold_volatility = st.sidebar.slider('Volatility Threshold Multiplier', min_value=1.0, max_value=2.0,
                                                 value=1.5, step=0.1)
        threshold_drawdown = st.sidebar.slider('Drawdown Threshold', min_value=0.01, max_value=0.10, value=0.05,
                                               step=0.01)

    elif analysis_mode == "Monte Carlo Simulation":
        n_simulations = st.sidebar.slider('Number of Simulations', min_value=100, max_value=5000, value=1000, step=100)
        n_days = st.sidebar.slider('Simulation Days', min_value=10, max_value=90, value=30, step=5)
        mc_confidence_level = st.sidebar.slider('Simulation Confidence Level', min_value=0.90, max_value=0.99,
                                                value=0.95, step=0.01)

    elif analysis_mode == "Feature Importance":
        importance_method = st.sidebar.multiselect(
            'Feature Importance Methods',
            ["F-test", "Mutual Information", "Random Forest", "XGBoost", "SHAP"],
            default=["F-test", "Mutual Information", "Random Forest", "XGBoost"]
        )

    elif analysis_mode == "Model Interpretation":
        shap_plot_type = st.sidebar.selectbox(
            'SHAP Plot Type',
            ["summary", "dependence", "waterfall"]
        )

    if st.sidebar.button('Run Analysis'):
        with st.spinner('Downloading data...'):
            try:
                api_key = "3POTM8L4ZKVIZW65"  # Note: In production, this should be retrieved from environment variables or secure storage
            except KeyError:
                st.error(
                    "API key not found. Please apply for a key (see ReadMe.md) and replace the api_key with your own key.")
                return

            data = download_stock_data_alpha_vantage(ticker, start_date, end_date, api_key)

        if data.empty:
            st.error('Unable to download data. Please check the stock symbol or date range.')
            return

        st.success('Data download successful!')

        # Display basic data information
        st.subheader('Data Overview')
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Number of Data Points", f"{len(data):,}")
        with col2:
            date_range = f"{data.index.min().date()} to {data.index.max().date()}"
            st.metric("Date Range", date_range)
        with col3:
            trading_days = len(data)
            calendar_days = (data.index.max() - data.index.min()).days
            coverage = trading_days / max(1, calendar_days) * 100
            st.metric("Data Coverage", f"{coverage:.1f}%")

        # Display raw data table
        with st.expander("View Raw Data"):
            st.dataframe(data.tail())

            # Calculate and display basic statistics
            st.write("Basic Statistics:")
            st.dataframe(data.describe())

        # Data preprocessing
        with st.spinner('Data preprocessing...'):
            processed_data = preprocess_data(data)

        st.success('Data preprocessing complete!')

        # Display preprocessed data
        with st.expander("View Preprocessed Data"):
            st.dataframe(processed_data.tail())
            st.write(f"Number of features after preprocessing: {len(processed_data.columns)}")

        # Check time series stationarity
        with st.expander("Time Series Stationarity Test"):
            adf_result, decomposition = check_stationarity(processed_data['Close'])

            st.write("ADF Test Results:")
            st.dataframe(pd.Series(adf_result))

            if adf_result['p-value'] < 0.05:
                st.success("Time series is stationary (p < 0.05)")
            else:
                st.warning("Time series is not stationary (p >= 0.05)")

            if decomposition is not None:
                st.write("Time Series Decomposition:")
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
                decomposition.observed.plot(ax=ax1)
                ax1.set_title('Observed')
                decomposition.trend.plot(ax=ax2)
                ax2.set_title('Trend')
                decomposition.seasonal.plot(ax=ax3)
                ax3.set_title('Seasonal')
                decomposition.resid.plot(ax=ax4)
                ax4.set_title('Residual')
                plt.tight_layout()
                st.pyplot(fig)

        # Plot technical indicators
        st.subheader('Technical Indicator Analysis')
        plot_advanced_technical_indicators(processed_data, ticker)

        # Execute different operations based on analysis mode
        if analysis_mode == "Basic Analysis":
            # Feature selection
            selected_features, importance_df = select_features(
                processed_data.drop('Close', axis=1),
                processed_data['Close'],
                n_features=20
            )

            st.subheader('Feature Importance')
            plot_feature_importance(importance_df)

            # Prepare dataset
            X, y = create_ml_dataset(processed_data, selected_features, target='Close')

            # Split training and test sets
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            st.write(f'Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}')

            # Train traditional models
            with st.spinner('Training machine learning models...'):
                ml_models, best_params, cv_scores, val_scores = build_and_train_advanced_ml_models(
                    X_train, y_train, X_test, y_test
                )

            st.success('Model training complete!')

            # Display model evaluation results
            st.subheader('Model Evaluation Results')

            # Create evaluation metrics DataFrame
            metrics_df = pd.DataFrame({
                'MSE (Cross-validation)': cv_scores,
                'MSE (Test set)': val_scores,
                'RMSE (Test set)': {k: np.sqrt(v) for k, v in val_scores.items()}
            })

            st.dataframe(metrics_df)

            # Find the best model
            best_model_name = metrics_df['MSE (Test set)'].idxmin()
            best_model = ml_models[best_model_name]

            st.write(f'Best model: **{best_model_name}**')

            # Display best model hyperparameters
            if best_model_name in best_params:
                st.write('Best model hyperparameters:')
                st.json(best_params[best_model_name])

            # Make predictions with the best model
            predictions = best_model.predict(X_test)

            # Create prediction Series
            test_dates = processed_data.index[split:split + len(predictions)]
            real_series = pd.Series(y_test.values, index=test_dates)
            pred_series = pd.Series(predictions, index=test_dates)

            # Plot prediction results
            st.subheader('Prediction Results')
            plot_predictions(real_series, pred_series, ticker)

            # Calculate prediction confidence interval
            residuals = y_test - predictions
            std_residuals = residuals.std()
            confidence_interval = {
                'lower': pred_series - 1.96 * std_residuals,
                'upper': pred_series + 1.96 * std_residuals
            }

            st.subheader('Prediction Results (with Confidence Interval)')
            plot_predictions(real_series, pred_series, ticker, confidence_interval)

        elif analysis_mode == "Advanced Analysis":
            # Feature selection
            if feature_selection:
                selected_features, importance_df = select_features(
                    processed_data.drop('Close', axis=1),
                    processed_data['Close'],
                    n_features=n_features
                )

                st.subheader('Feature Importance')
                plot_feature_importance(importance_df)
            else:
                # Use all features
                selected_features = processed_data.columns.tolist()
                if 'Close' in selected_features:
                    selected_features.remove('Close')

            # Prepare data for deep learning
            dataset = processed_data[['Close'] + selected_features].values

            # Normalize data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)

            # Create time series dataset
            X_seq, y_seq = create_lstm_dataset(scaled_data, time_step)

            # Split training and test sets
            split_idx = int(len(X_seq) * 0.8)
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

            # Further split validation set
            val_idx = int(len(X_train) * 0.8)
            X_val = X_train[val_idx:]
            y_val = y_train[val_idx:]
            X_train = X_train[:val_idx]
            y_train = y_train[:val_idx]

            st.write(
                f'Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}, Test set size: {X_test.shape[0]}')

            # Train model based on selected model type
            with st.spinner(f'Training {model_type} model...'):
                if model_type == "LSTM":
                    dl_model, history = build_and_train_advanced_lstm(
                        X_train, y_train, X_val, y_val, epochs, batch_size, patience
                    )
                elif model_type == "Bidirectional LSTM":
                    dl_model, history = build_and_train_advanced_lstm(
                        X_train, y_train, X_val, y_val, epochs, batch_size, patience
                    )
                elif model_type == "GRU":
                    dl_model, history = build_and_train_gru(
                        X_train, y_train, X_val, y_val, epochs, batch_size, patience
                    )
                elif model_type == "CNN-LSTM":
                    dl_model, history = build_and_train_cnn_lstm(
                        X_train, y_train, X_val, y_val, epochs, batch_size, patience
                    )

            st.success(f'{model_type} model training complete!')

            # Plot training history
            st.subheader('Model Training History')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            ax1.plot(history.history['loss'], label='Training Loss')
            ax1.plot(history.history['val_loss'], label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()

            if 'lr' in history.history:
                ax2.plot(history.history['lr'], label='Learning Rate')
                ax2.set_title('Learning Rate Change')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Learning Rate')
                ax2.legend()

            st.pyplot(fig)

            # Model prediction
            predictions = dl_model.predict(X_test)

            # Inverse transform predictions
            predictions_dummy = np.zeros((len(predictions), dataset.shape[1]))
            predictions_dummy[:, 0] = predictions.flatten()
            predictions_inverse = scaler.inverse_transform(predictions_dummy)[:, 0]

            # Inverse transform actual values
            y_test_dummy = np.zeros((len(y_test), dataset.shape[1]))
            y_test_dummy[:, 0] = y_test
            y_test_inverse = scaler.inverse_transform(y_test_dummy)[:, 0]

            # Create prediction Series
            test_dates = processed_data.index[split_idx + time_step:split_idx + time_step + len(predictions)]
            real_series = pd.Series(y_test_inverse, index=test_dates)
            pred_series = pd.Series(predictions_inverse, index=test_dates)

            # Plot prediction results
            st.subheader('Prediction Results')
            plot_predictions(real_series, pred_series, ticker)

            # Calculate evaluation metrics
            mse = mean_squared_error(y_test_inverse, predictions_inverse)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_inverse, predictions_inverse)
            mape = mean_absolute_percentage_error(y_test_inverse, predictions_inverse)
            r2 = r2_score(y_test_inverse, predictions_inverse)

            # Display evaluation metrics
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("MSE", f"{mse:.4f}")
            with col2:
                st.metric("RMSE", f"{rmse:.4f}")
            with col3:
                st.metric("MAE", f"{mae:.4f}")
            with col4:
                st.metric("MAPE", f"{mape:.4%}")
            with col5:
                st.metric("R²", f"{r2:.4f}")

        elif analysis_mode == "Risk Assessment":
            # Feature selection
            selected_features, importance_df = select_features(
                processed_data.drop('Close', axis=1),
                processed_data['Close'],
                n_features=20
            )

            # Prepare dataset
            X, y = create_ml_dataset(processed_data, selected_features, target='Close')

            # Split training and test sets
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Train model
            with st.spinner('Training model...'):
                model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
                model.fit(X_train, y_train)

            # Predict
            predictions = model.predict(X_test)

            # Create prediction Series
            test_dates = processed_data.index[split:split + len(predictions)]
            real_series = pd.Series(y_test.values, index=test_dates)
            pred_series = pd.Series(predictions, index=test_dates)

            # Plot prediction results
            st.subheader('Prediction Results')
            plot_predictions(real_series, pred_series, ticker)

            # Advanced risk assessment
            st.subheader('Advanced Risk Assessment')
            risk_df = assess_risk_advanced(
                real_series, pred_series,
                window=window,
                threshold_volatility=threshold_volatility,
                threshold_drawdown=threshold_drawdown,
                confidence_level=confidence_level
            )

            # Display risk assessment results
            st.write("Risk Assessment Results:")
            st.dataframe(risk_df[['Real', 'Predicted', 'Error', 'Percent_Error', 'Volatility', 'Drawdown', 'VaR',
                                  'Risk_Level']].tail())

            # Risk level distribution
            st.write("Risk Level Distribution:")
            risk_counts = risk_df['Risk_Level'].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title='Risk Level Distribution',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig)

            # Plot risk heatmap
            st.subheader('Risk Heatmap')
            plot_risk_heatmap(risk_df, ticker)

            # Calculate VaR and CVaR
            returns = processed_data['Daily_Return'].dropna()
            var_results, cvar_results = calculate_var_cvar(returns, confidence_levels=[0.95, 0.99])

            # Display VaR and CVaR
            st.subheader('Value at Risk (VaR) and Conditional Value at Risk (CVaR)')

            col1, col2 = st.columns(2)

            with col1:
                st.write("Value at Risk (VaR):")
                for level, value in var_results.items():
                    st.metric(f"{level}", f"{value:.4%}")

            with col2:
                st.write("Conditional Value at Risk (CVaR):")
                for level, value in cvar_results.items():
                    st.metric(f"{level}", f"{value:.4%}")

            # Risk time series plot
            st.subheader('Risk Time Series')

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=risk_df.index,
                y=risk_df['Volatility'],
                mode='lines',
                name='Volatility'
            ))

            fig.add_trace(go.Scatter(
                x=risk_df.index,
                y=risk_df['Drawdown'],
                mode='lines',
                name='Drawdown'
            ))

            fig.add_trace(go.Scatter(
                x=risk_df.index,
                y=abs(risk_df['VaR']),
                mode='lines',
                name='VaR (Absolute Value)'
            ))

            # Set background color based on risk level
            for i, risk_level in enumerate(risk_df['Risk_Level'].unique()):
                mask = risk_df['Risk_Level'] == risk_level
                if mask.any():
                    risk_periods = []
                    start_idx = None

                    for idx, is_risk in enumerate(mask):
                        if is_risk and start_idx is None:
                            start_idx = idx
                        elif not is_risk and start_idx is not None:
                            risk_periods.append((start_idx, idx - 1))
                            start_idx = None

                    if start_idx is not None:
                        risk_periods.append((start_idx, len(mask) - 1))

                    for start, end in risk_periods:
                        color = None
                        if risk_level == 'Low':
                            color = 'rgba(0, 255, 0, 0.1)'
                        elif risk_level == 'Medium':
                            color = 'rgba(255, 255, 0, 0.1)'
                        elif risk_level == 'High':
                            color = 'rgba(255, 165, 0, 0.1)'
                        elif risk_level == 'Very High':
                            color = 'rgba(255, 0, 0, 0.1)'

                        if color:
                            fig.add_vrect(
                                x0=risk_df.index[start],
                                x1=risk_df.index[end],
                                fillcolor=color,
                                opacity=0.5,
                                layer="below",
                                line_width=0,
                                annotation_text=risk_level,
                                annotation_position="top left"
                            )

            fig.update_layout(
                title=f'{ticker} Risk Indicators Time Series',
                xaxis_title='Date',
                yaxis_title='Risk Indicator Value',
                template='plotly_white',
                hovermode='x unified'
            )

            st.plotly_chart(fig)

        elif analysis_mode == "Monte Carlo Simulation":
            # Perform Monte Carlo simulation
            st.subheader('Monte Carlo Simulation')

            with st.spinner(f'Performing {n_simulations} Monte Carlo simulations...'):
                sim_df, daily_var = monte_carlo_simulation(
                    processed_data,
                    n_simulations=n_simulations,
                    n_days=n_days,
                    confidence_level=mc_confidence_level
                )

            st.success('Monte Carlo simulation complete!')

            # Plot Monte Carlo simulation results
            last_price = processed_data['Close'].iloc[-1]
            plot_monte_carlo(sim_df, ticker, last_price, confidence_level=mc_confidence_level)

            # Display statistics
            st.subheader('Simulation Statistics')

            # Final day price distribution
            final_prices = sim_df.iloc[-1]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Average Forecast Price", f"{final_prices.mean():.2f}")
            with col2:
                st.metric("Minimum Forecast Price", f"{final_prices.min():.2f}")
            with col3:
                st.metric("Maximum Forecast Price", f"{final_prices.max():.2f}")

            # Plot final day price distribution
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=final_prices,
                nbinsx=50,
                marker_color='blue',
                opacity=0.7
            ))

            # Add current price line
            fig.add_vline(
                x=last_price,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Current Price: {last_price:.2f}",
                annotation_position="top right"
            )

            # Add VaR line
            var_price = np.percentile(final_prices, (1 - mc_confidence_level) * 100)
            fig.add_vline(
                x=var_price,
                line_dash="dash",
                line_color="green",
                annotation_text=f"{int(mc_confidence_level * 100)}% VaR: {var_price:.2f}",
                annotation_position="top left"
            )

            fig.update_layout(
                title=f'Price Distribution after {n_days} days',
                xaxis_title='Price',
                yaxis_title='Frequency',
                template='plotly_white'
            )

            st.plotly_chart(fig)

            # Plot daily VaR
            var_series = pd.Series(daily_var)

            fig = px.line(
                x=var_series.index,
                y=var_series.values,
                title=f'{int(mc_confidence_level * 100)}% Daily VaR',
                labels={'x': 'Days', 'y': 'VaR'}
            )

            st.plotly_chart(fig)

            # Calculate return distribution
            returns = (final_prices - last_price) / last_price

            # Calculate statistics
            mean_return = returns.mean()
            std_return = returns.std()
            skew = returns.skew()
            kurt = returns.kurtosis()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Average Return", f"{mean_return:.2%}")
            with col2:
                st.metric("Return Standard Deviation", f"{std_return:.2%}")
            with col3:
                st.metric("Skewness", f"{skew:.4f}")
            with col4:
                st.metric("Kurtosis", f"{kurt:.4f}")

            # Plot return distribution
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                marker_color='green',
                opacity=0.7
            ))

            # Add zero return line
            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="red",
                annotation_text="Zero Return",
                annotation_position="top right"
            )

            fig.update_layout(
                title=f'Return Distribution after {n_days} days',
                xaxis_title='Return',
                yaxis_title='Frequency',
                template='plotly_white'
            )

            st.plotly_chart(fig)

            # Calculate profit probability
            profit_prob = (returns > 0).mean()
            significant_profit_prob = (returns > 0.05).mean()
            loss_prob = (returns < 0).mean()
            significant_loss_prob = (returns < -0.05).mean()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Profit Probability", f"{profit_prob:.2%}")
            with col2:
                st.metric("Significant Profit Probability (>5%)", f"{significant_profit_prob:.2%}")
            with col3:
                st.metric("Loss Probability", f"{loss_prob:.2%}")
            with col4:
                st.metric("Significant Loss Probability (>5%)", f"{significant_loss_prob:.2%}")

        elif analysis_mode == "Feature Importance":
            # Feature importance analysis
            st.subheader('Feature Importance Analysis')

            # Prepare data
            X = processed_data.drop('Close', axis=1)
            y = processed_data['Close']

            # Calculate various feature importances
            importance_methods = {}

            if "F-test" in importance_method:
                from sklearn.feature_selection import f_regression
                f_values, _ = f_regression(X, y)
                importance_methods['F-test'] = pd.Series(f_values, index=X.columns)

            if "Mutual Information" in importance_method:
                from sklearn.feature_selection import mutual_info_regression
                mi_values = mutual_info_regression(X, y)
                importance_methods['Mutual Information'] = pd.Series(mi_values, index=X.columns)

            if "Random Forest" in importance_method:
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                importance_methods['Random Forest'] = pd.Series(rf.feature_importances_, index=X.columns)

            if "XGBoost" in importance_method:
                import xgboost as xgb
                xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                xgb_model.fit(X, y)
                importance_methods['XGBoost'] = pd.Series(xgb_model.feature_importances_, index=X.columns)

            if "SHAP" in importance_method:
                import xgboost as xgb
                import shap

                with st.spinner('Calculating SHAP values...'):
                    # Use XGBoost model
                    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                    model.fit(X, y)

                    # Calculate SHAP values
                    explainer = shap.Explainer(model)
                    shap_values = explainer(X)

                    # Calculate mean absolute SHAP values
                    shap_importance = pd.Series(np.abs(shap_values.values).mean(0), index=X.columns)
                    importance_methods['SHAP'] = shap_importance

                # Plot SHAP summary plot
                st.subheader('SHAP Summary Plot')
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X, plot_type="bar")
                st.pyplot(plt)

                # Plot SHAP waterfall plot
                st.subheader('SHAP Waterfall Plot (Random Sample)')
                plt.figure(figsize=(10, 8))
                sample_idx = np.random.randint(0, len(X))
                shap.plots.waterfall(shap_values[sample_idx])
                st.pyplot(plt)

            # Merge all feature importances
            all_importance = pd.DataFrame(importance_methods)

            # Normalize feature importances
            for col in all_importance.columns:
                all_importance[f'{col}_norm'] = (all_importance[col] - all_importance[col].min()) / (
                            all_importance[col].max() - all_importance[col].min())

            # If more than one method, calculate average importance
            if len(importance_method) > 1:
                norm_cols = [f'{col}_norm' for col in all_importance.columns if col in importance_method]
                all_importance['Average Importance'] = all_importance[norm_cols].mean(axis=1)
                all_importance = all_importance.sort_values('Average Importance', ascending=False)
            else:
                # Use single method for sorting
                sort_col = list(importance_methods.keys())[0]
                all_importance = all_importance.sort_values(sort_col, ascending=False)

            # Display feature importance table
            st.write("Feature Importance Table:")
            st.dataframe(all_importance)

            # Plot feature importance bar chart
            st.subheader('Feature Importance Visualization')

            # Select top 15 features
            top_features = all_importance.index[:15]
            top_importance = all_importance.loc[top_features]

            # Create bar chart
            fig = go.Figure()

            for method in importance_method:
                if method in top_importance.columns:
                    fig.add_trace(go.Bar(
                        y=top_features,
                        x=top_importance[method],
                        name=method,
                        orientation='h'
                    ))

            fig.update_layout(
                title='Feature Importance Comparison',
                xaxis_title='Importance Score',
                yaxis_title='Feature',
                barmode='group',
                height=600,
                template='plotly_white'
            )

            st.plotly_chart(fig)

            # Feature correlation analysis
            st.subheader('Feature Correlation Analysis')

            # Calculate correlation matrix
            corr_matrix = processed_data.corr()

            # Plot heatmap
            fig = px.imshow(
                corr_matrix,
                title='Feature Correlation Heatmap',
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )

            fig.update_layout(
                height=800,
                width=800
            )

            st.plotly_chart(fig)

            # Correlation with target variable
            target_corr = corr_matrix['Close'].sort_values(ascending=False)

            st.write("Features with highest correlation to Close price:")
            st.dataframe(target_corr)

            # Plot bar chart of target correlations
            fig = px.bar(
                x=target_corr.index,
                y=target_corr.values,
                title='Correlation with Close Price',
                color=target_corr.values,
                color_continuous_scale='RdBu_r',
                range_color=[-1, 1]
            )

            fig.update_layout(
                xaxis_title='Feature',
                yaxis_title='Correlation Coefficient',
                xaxis_tickangle=-45,
                height=500
            )

            st.plotly_chart(fig)



        elif analysis_mode == "Model Interpretation":

            # Model interpretation analysis

            st.subheader('Model Interpretation Analysis')

            # Add these lines to import both required modules

            import xgboost as xgb

            import shap

            # Prepare data

            X = processed_data.drop('Close', axis=1)

            y = processed_data['Close']

            # Train XGBoost model

            with st.spinner('Training XGBoost model...'):

                model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

                model.fit(X, y)

            st.success('Model training complete!')

            # Use SHAP for model interpretation

            with st.spinner('Calculating SHAP values...'):

                explainer = shap.Explainer(model)

                shap_values = explainer(X)

            st.success('SHAP values calculation complete!')

            # Plot SHAP plot based on selected type
            if shap_plot_type == "summary":
                st.write(
                    "SHAP summary plot shows the magnitude and direction of each feature's impact on model predictions.")
                plt.figure(figsize=(12, 10))
                shap.summary_plot(shap_values, X)
                st.pyplot(plt)

                plt.figure(figsize=(12, 10))
                shap.summary_plot(shap_values, X, plot_type="bar")
                st.pyplot(plt)

            elif shap_plot_type == "dependence":
                st.write("SHAP dependence plot shows how a specific feature's SHAP values change with feature values.")

                # Find the most important features
                mean_shap = np.abs(shap_values.values).mean(0)
                top_features = X.columns[np.argsort(mean_shap)[-5:]]

                for feature in top_features:
                    plt.figure(figsize=(12, 8))
                    shap.dependence_plot(feature, shap_values.values, X, feature_names=X.columns)
                    st.pyplot(plt)

            elif shap_plot_type == "waterfall":
                st.write("SHAP waterfall plot shows the contribution of each feature to a single prediction.")

                # Allow user to select a sample
                sample_index = st.slider("Select sample index", 0, len(X) - 1, 0)

                plt.figure(figsize=(12, 8))
                shap.plots.waterfall(shap_values[sample_index])
                st.pyplot(plt)

                # Display selected sample's actual feature values
                st.write("Selected sample feature values:")
                sample_df = X.iloc[sample_index].to_frame().reset_index()
                sample_df.columns = ['Feature', 'Value']
                st.dataframe(sample_df)

            # Add model interpretation conclusions
            st.subheader('Model Interpretation Conclusions')

            # Calculate feature importance
            mean_shap = pd.Series(np.abs(shap_values.values).mean(0), index=X.columns)
            top_features = mean_shap.sort_values(ascending=False).head(5).index.tolist()

            st.write(
                f"Based on SHAP value analysis, the 5 most important factors for predicting {ticker} stock price are:")
            for i, feature in enumerate(top_features, 1):
                st.write(f"{i}. **{feature}** (Importance score: {mean_shap[feature]:.4f})")

            # Calculate feature effect direction
            feature_effects = {}
            for feature in top_features:
                feature_values = X[feature]
                feature_shap = shap_values[:, feature]

                # Calculate positive and negative impacts
                pos_impact = ((feature_shap.values > 0) & (feature_values > feature_values.mean())).sum() + \
                             ((feature_shap.values < 0) & (feature_values < feature_values.mean())).sum()
                neg_impact = ((feature_shap.values < 0) & (feature_values > feature_values.mean())).sum() + \
                             ((feature_shap.values > 0) & (feature_values < feature_values.mean())).sum()

                # Determine primary effect direction
                if pos_impact > neg_impact:
                    direction = "Positive relationship"
                else:
                    direction = "Negative relationship"

                feature_effects[feature] = direction

            st.write("Relationship of these factors with stock price:")
            for feature, direction in feature_effects.items():
                st.write(f"- **{feature}**: {direction}")

    # Add author information
    st.sidebar.markdown("---")
    st.sidebar.info("Authors: Du Hongzhou & Qi Legan (Team: Studio 6324)")


if __name__ == "__main__":
    main()