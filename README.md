# 🚗 Used Car Price Predictor

A Machine Learning web application that predicts the resale value of cars based on features like brand, model, age, mileage, and horsepower.

## 🎯 Project Overview
* **Problem:** Accurately estimating used car prices is difficult due to the high number of variables.
* **Solution:** A Random Forest Regressor model trained on market data to predict prices with high accuracy.
* **Key Feature:** Uses "Smart Grouping" to handle over 500+ car models by prioritizing the top 20 most popular ones.

## 🛠️ Tech Stack
* **Python** (Logic & Data Processing)
* **Scikit-Learn** (Machine Learning: Random Forest)
* **Streamlit** (Web Interface)
* **Pandas & NumPy** (Data Manipulation)

## 🚀 How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/snigdhasirivalli/used_cars_price_prediction.git](https://github.com/snigdhasirivalli/used_cars_price_prediction.git)
    cd used_cars_price_prediction
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the Model** (Optional - Model is already included)
    ```bash
    python train.py
    ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure
* `train.py`: The training pipeline. It cleans data, engineers features, trains the model, and saves it as a .joblib file.
* `app.py`: The frontend application. It loads the saved model and accepts user input for real-time predictions.
* `requirements.txt`: List of required Python libraries.