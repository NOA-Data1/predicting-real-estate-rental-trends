# Real Estate Market Forecasting using Deep Neural Networks 🚀

## 📖 Project Overview
This project implements a predictive pipeline to forecast rental price trends using **Deep Learning**. By comparing **RNN** and **LSTM** architectures, the model identifies long-term patterns in the real estate market to provide reliable 24 and 48-month projections.

---

## 🧠 Core Topics & Technical Features

### 1. Deep Learning Architectures
I explored two main types of Recurrent Neural Networks:
* **Simple RNN:** Used as a baseline to capture sequential data.
* **LSTM (Long Short-Term Memory):** Implemented to solve the vanishing gradient problem, allowing the model to remember rental trends from several years back.



### 2. Data Engineering & Preprocessing
To ensure the Neural Networks performed accurately, the following steps were taken:
* **Feature Scaling:** Applied `StandardScaler` to normalize Euro values, preventing numerical instability.
* **Time-Series Windowing:** Transformed raw data into a supervised learning format using "look-back" windows.
* **Handling Missing Data:** Cleaned the dataset to ensure continuous time-series sequences.

### 3. Model Optimization
* **Early Stopping:** A crucial callback that monitors validation loss and stops training at the optimal point to prevent **Overfitting**.
* **Dropout Layers:** Integrated to improve the model's ability to generalize to new, unseen market data.



### 4. Market Forecasting (Results)
The models provide two strategic horizons:
* **24-Month Forecast:** Captures immediate market volatility and seasonality.
* **48-Month Forecast:** Identifies long-term stabilization trends, useful for real estate investment planning.

---

## 🛠️ Technology Stack
* **Language:** Python (VS Code)
* **Deep Learning:** TensorFlow, Keras
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn

---

## ⚙️ How to Use
1. Clone the repository.
2. Ensure you have the required libraries: `pip install tensorflow pandas scikit-learn matplotlib`.
3. Run the script: `python real-estate-neural-network-forecasting.py`.
