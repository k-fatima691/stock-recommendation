from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

app = Flask(__name__)

# Load the trained model and data (scaled data, pca data, clustering model)
model_data = joblib.load('stock_recommendation_model.pkl')

# Extract relevant data from the model
scaled_data = model_data['scaled_data']
pca_data = model_data['pca_data']
model = model_data['model']
daily_returns = model_data['daily_returns']  # Contains the 'Cluster' column after clustering

# Function for stock recommendations
def recommend_stocks(daily_returns, input_ticker, top_n=3):
    # Step 1: Match the correct column name in daily_returns
    matching_columns = [col for col in daily_returns.columns if input_ticker in col]

    if not matching_columns:
        return f"Error: {input_ticker} not found in daily_returns."

    # Use the first match (assuming 'Adj Close' is always part of the format)
    input_column = matching_columns[0]

    # Step 2: Compute correlations with other stocks
    correlations = {}
    for column in daily_returns.columns:
        # Ensure it is an 'Adj Close' column and not the input stock itself
        if 'Adj Close' in column and column != input_column:
            stock = column.split('_')[0]  # Extract the stock ticker (e.g., 'APL.KA')
            corr = daily_returns[input_column].corr(daily_returns[column])
            correlations[stock] = corr

    # Step 3: Sort stocks by correlation in descending order
    sorted_stocks = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    # Step 4: Return top N recommendations
    recommendations = sorted_stocks[:top_n]
    return recommendations

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/recommend', methods=['POST'])
def recommend():
    ticker = request.form.get('ticker')
    top_n = int(request.form.get('top_n', 3))  # Default to 3 if not specified
    
    if not ticker:
        return "No ticker entered!", 400  # Return error message if no ticker is entered

    recommendations = recommend_stocks(daily_returns, ticker, top_n)
    
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
