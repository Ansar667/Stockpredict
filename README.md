# Stockpredict

Stockpredict is a small end-to-end project for stock price prediction.  
It combines a machine learning model written in Python with a simple web interface so that users can request a prediction through the browser.

> ⚠️ Disclaimer: This project is for learning and educational purposes only and **not** financial advice.

---

## Features

- 📈 **Stock price prediction** using a trained ML model  
- 🧹 **Data preprocessing** and utility functions for working with time series  
- 🌐 **Web interface** to input a stock and see the prediction in the browser  
- 🧪 **Test script** to quickly check predictions from the command line  
- 📦 **Separated modules** for model, training, utilities and web app

---

## Tech Stack

- **Language:** Python  
- **Machine Learning:** scikit-learn / time series tools (see `requirements.txt`)  
- **Web:** simple Python web app (HTML, CSS, JavaScript in `templates/` and `static/`)  
- **Version control:** Git & GitHub

---

## Project Structure

```text
Stockpredict/
├── app.py             # Entry point for the web application
├── ml_utils.py        # Helper functions for preprocessing and prediction
├── model.py           # Model class / functions used by the app
├── train_models.py    # Script to train and save the ML model(s)
├── test_predict.py    # Simple script to test model predictions
├── requirements.txt   # Python dependencies
├── models/            # Saved model files
├── static/            # CSS, JS and other static assets for the UI
├── templates/         # HTML templates for the web interface
└── __pycache__/       # Python cache (can be ignored)
How It Works (Overview)
Training

train_models.py loads historical stock price data,

preprocesses it (e.g. scaling, splitting into train/test),

trains a model and saves it into the models/ directory.

Prediction logic

model.py and ml_utils.py contain the code for loading the saved model

and generating predictions for new input data.

Web app

app.py starts a web server and connects the UI with the model.

HTML templates in templates/ and static files in static/ are used to render the page.

The user enters parameters (e.g. a stock / dates), the backend returns the prediction and shows it in the browser.

CLI testing

test_predict.py can be used to quickly check that the model and utilities work as expected without running the web app.

Installation
Clone the repository

bash
Копировать код
git clone https://github.com/Ansar667/Stockpredict.git
cd Stockpredict
Create and activate a virtual environment (optional, but recommended)

bash
Копировать код
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
Install dependencies

bash
Копировать код
pip install -r requirements.txt
Usage
1. Train the model
If you haven’t trained the model yet or changed the code, run:

bash
Копировать код
python train_models.py
This will create/update the model files inside the models/ folder.

2. Run the web application
bash
Копировать код
python app.py
After that, open the URL printed in the terminal
(usually something like http://127.0.0.1:5000/) in your browser.

3. Test predictions from the command line (optional)
bash
Копировать код
python test_predict.py
This script is useful to quickly check that the model is loading correctly and returning a prediction.

Roadmap / Possible Improvements
Add more advanced time series models (e.g. LSTM, Prophet, etc.)

Add data visualization (plots of historical vs predicted prices)

Support multiple tickers and more flexible user input

Dockerfile for easier deployment

Deploy the app to a cloud platform (Render, Railway, etc.)

