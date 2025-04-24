# Financial Risk Assessment and Early Warning System

![GitHub stars](https://img.shields.io/github/stars/DwHz/Financial_Risk_Assessment-Early_Warning_System?style=social)
![GitHub forks](https://img.shields.io/github/forks/DwHz/Financial_Risk_Assessment-Early_Warning_System?style=social)
![GitHub issues](https://img.shields.io/github/issues/DwHz/Financial_Risk_Assessment-Early_Warning_System)
![GitHub license](https://img.shields.io/github/license/DwHz/Financial_Risk_Assessment-Early_Warning_System)

**Authors: Hongzhou Du**


## 📊 Project Overview

The Financial Risk Assessment and Early Warning System is a sophisticated analysis tool built on machine learning and deep learning technologies. It provides comprehensive capabilities for stock market data analysis, future price prediction, and risk assessment through an interactive Streamlit interface. This system is designed for investors, financial analysts, and quantitative researchers who need advanced tools for market analysis and decision support.

### Purpose

This system addresses the challenge of financial risk prediction by:
- Analyzing historical stock data using multiple technical indicators
- Employing advanced machine learning models to predict future price movements
- Assessing potential risk levels through multi-dimensional analysis
- Providing interactive visualizations for better decision-making
- Supporting risk quantification through VaR and other metrics

## ✨ Key Features

### 🔍 Data Acquisition and Processing
- **Historical Data Retrieval**: Downloads stock data via Alpha Vantage API
- **Technical Indicator Calculation**: 
  - Multiple timeframe RSI (7, 14, 21 days)
  - MACD with signal and histogram
  - Bollinger Bands (multiple periods)
  - Stochastic Oscillator
  - ADX (Average Directional Index)
  - ATR (Average True Range)
  - Volume indicators (OBV, VWAP, Chaikin Money Flow)
  - Ichimoku Cloud
  - CCI (Commodity Channel Index)
  - Many more trend, momentum and volatility indicators
- **Advanced Data Preprocessing**:
  - Missing value handling
  - Feature scaling
  - Feature engineering
  - Time series stationarity testing

### 🧠 Prediction Models
- **Deep Learning Models**:
  - LSTM (Long Short-Term Memory)
  - Bidirectional LSTM
  - GRU (Gated Recurrent Units)
  - CNN-LSTM hybrid
- **Traditional Machine Learning Models**:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Support Vector Regression
  - XGBoost
  - AdaBoost
  - Voting Regressor (ensemble)
- **Hyperparameter Tuning**: Automated optimization via GridSearchCV with time series cross-validation

### ⚠️ Risk Assessment
- **Multi-dimensional Risk Quantification**:
  - Volatility-based risk assessment
  - Drawdown analysis
  - Error-based risk evaluation
  - Value at Risk (VaR) calculation
- **Comprehensive Risk Levels**: Low, Medium, High, Very High classifications
- **Advanced Metrics**:
  - VaR at multiple confidence levels (95%, 99%)
  - Conditional Value at Risk (CVaR/Expected Shortfall)
  - Risk heatmaps and time series

### 📈 Monte Carlo Simulation
- Price path simulation (customizable number of simulations)
- Future price distribution analysis
- Probability of profit/loss calculation
- Daily VaR calculation through simulation

### 💡 Model Interpretation
- **Feature Importance Analysis**:
  - F-test
  - Mutual Information
  - Random Forest importance
  - XGBoost importance
- **SHAP (SHapley Additive exPlanations)**:
  - Summary plots
  - Dependence plots
  - Waterfall plots
- **Feature correlation analysis**

### 🎯 Advanced Visualization
- **Interactive Charts**:
  - Candlestick charts with technical indicators
  - Prediction results with confidence intervals
  - Risk heatmaps
  - Monte Carlo simulation paths
  - Feature importance visualization
  - SHAP value plots
  - Technical indicator dashboards

## 🚀 Installation Guide

### Prerequisites
- Python 3.7+
- pip (Python package manager)
- Git (for cloning the repository)

### Step 1: Clone the Repository
```bash
git clone https://github.com/DwHz/Financial_Risk_Assessment-Early_Warning_System.git
cd Financial_Risk_Assessment-Early_Warning_System
```

### Step 2: Create a Virtual Environment (recommended)
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Libraries
```bash
pip install -r requirements.txt
```

### Step 4: Alpha Vantage API Key
1. Obtain an Alpha Vantage API key from [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
2. Register for a free account
3. After registration, you'll receive a free API key
4. The system includes a default API key, but consider replacing it with your own key for better performance

## 📝 Usage Instructions

### Starting the Application
```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

### Using the Application

#### 1. Setup Parameters
In the sidebar:
- Enter a stock symbol (e.g., "AAPL" for Apple)
- Select start and end dates for analysis
- Choose an analysis mode:
  - Basic Analysis
  - Advanced Analysis
  - Risk Assessment
  - Monte Carlo Simulation
  - Feature Importance
  - Model Interpretation
- Adjust model parameters based on your selected mode

#### 2. Run Analysis
- Click the "Run Analysis" button to start processing
- The system will download data, calculate indicators, train models, and display results

#### 3. Explore Results
Each analysis mode provides different insights:

##### Basic Analysis
- Data overview with basic statistics
- Technical indicator visualization
- Feature importance analysis
- Machine learning model evaluation
- Prediction results with confidence intervals

##### Advanced Analysis
- Deep learning model training and evaluation
- Visualization of model training history
- Prediction results with comprehensive metrics
- Model performance comparison

##### Risk Assessment
- Advanced risk evaluation across multiple dimensions
- Risk level distribution
- Risk heatmap visualization
- VaR and CVaR calculation
- Risk time series analysis

##### Monte Carlo Simulation
- Price path simulation visualization
- Final price distribution analysis
- Return probabilities and statistics
- Daily VaR through simulation

##### Feature Importance
- Feature importance comparison across multiple methods
- Correlation analysis
- Feature-to-target relationship visualization

##### Model Interpretation
- SHAP value analysis for model interpretability
- Feature impact visualization
- Model interpretation conclusions


### Technical Indicator Analysis
The system calculates and visualizes multiple technical indicators, including:
- Moving averages (multiple periods)
- RSI, MACD, and Stochastic oscillators
- Bollinger Bands
- Ichimoku Cloud
- ADX, ATR, and more

### Prediction Results
Models are trained on historical data and used to predict future prices. Results are visualized with:
- Actual vs. predicted price charts
- Confidence intervals
- Performance metrics (MSE, RMSE, MAE, MAPE, R²)

### Risk Assessment
The system provides comprehensive risk analysis with:
- Multi-dimensional risk level classification
- Risk heatmap visualization
- VaR and CVaR calculations
- Time series risk indicators

## 🧰 Adding Custom Features

The system is designed to be extensible. To add new features:

### Adding New Technical Indicators
1. Create a function in `app.py` to calculate the indicator
2. Add the indicator calculation to the `preprocess_data` function
3. Update visualization functions to display the new indicator

### Adding New Models
1. Create a function for your model architecture
2. Add your model to the relevant model selection section
3. Update training and evaluation code to support the new model


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📞 Contact

- Hongzhou Du - [duhz2929@gmail.com](duhz2929@gmail.com)
