name : Devatha Jayanth Machineni
id:CT08PP682
domain: Data Science
duration: May 20 - June 20
mentor: Sravani Gouni
description: 
**Linear Regression Model for Housing Data**

This project demonstrates the implementation of a simple linear regression model using a housing dataset, specifically focusing on predicting the median house value (`MEDV`) based on the average number of rooms per dwelling (`RM`). The dataset is sourced from the HousingData.csv file stored locally on the user's machine. The main objective is to build a predictive model, evaluate its performance, and visualize the results in an intuitive and aesthetically pleasing manner using a web interface.

The project is built using Python and Flask, a lightweight web framework. The dataset is preprocessed and split into training and testing sets using `train_test_split` from `sklearn.model_selection`. The linear regression model is created and trained using `LinearRegression` from `sklearn.linear_model`. The model's performance is evaluated using two key metrics: Mean Squared Error (MSE) and R-squared (R²). These metrics are displayed on the main web page for easy assessment of the model's accuracy.

Visualization is a crucial part of this project. The actual versus predicted values are plotted using Plotly, a powerful graphing library, and the interactive plot is embedded into the webpage. This plot helps in understanding the model's performance at a glance by showing how closely the predicted values align with the actual values.

The web interface is designed for clarity and ease of navigation. The main page displays the evaluation metrics and visualization. Additionally, there are two buttons for navigating to separate pages that display the training data and the test data with predictions. These tables are styled with CSS to ensure readability and a professional look. The project emphasizes responsive design, ensuring that the content is accessible and visually appealing across different devices.

Overall, this project not only highlights the process of building a linear regression model but also showcases how to deploy it in a real-time web application, making data science results accessible to a broader audience.
