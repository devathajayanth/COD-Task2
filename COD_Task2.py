from flask import Flask, render_template
import pandas as pd
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

@app.route('/')
def index():
    # Load the dataset
    data = pd.read_csv(r'D:\COD_TASK2\HousingData.csv')

    # Separate features (X) and target variable (y)
    X = data[['RM']]  # Using only one feature for simplicity
    y = data['MEDV']  # Target variable

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    # Convert predictions to integers for better readability
    y_pred_int = y_pred.astype(int)

    # Visualize the regression line and actual vs. predicted values
    plot_div = visualize_regression(X_test, y_test, y_pred)

    return render_template('index.html', mse=mse, r_squared=r_squared, plot_div=plot_div,
                           X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, y_pred_int=y_pred_int)

def visualize_regression(X_test, y_test, y_pred):
    trace1 = go.Scatter(x=X_test['RM'], y=y_test, mode='markers', name='Actual', marker=dict(color='blue'))
    trace2 = go.Scatter(x=X_test['RM'], y=y_pred, mode='lines', name='Predicted', line=dict(color='red'))

    layout = go.Layout(title='Actual vs. Predicted Values',
                       xaxis=dict(title='Average Number of Rooms (RM)'),
                       yaxis=dict(title='Median House Value (MEDV)'))

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    plot_div = fig.to_html(full_html=False)
    return plot_div

if __name__ == '__main__':
    app.run(debug=True)
