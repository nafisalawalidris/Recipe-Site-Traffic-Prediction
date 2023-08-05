# Recipe Site Traffic Prediction

Welcome to the Recipe Site Traffic Prediction project repository. This project aims to predict high traffic recipes on a recipe website using machine learning techniques. The predictions will help the website's product manager make data-driven decisions to improve user engagement and overall traffic on the website.

## Project Overview
The goal of this project is to build a predictive model that can accurately identify recipes that are likely to generate high traffic. By leveraging historical data on recipe features and traffic patterns, we can train a machine learning model to predict the likelihood of a recipe being popular and generating high traffic.

## Data Validation and Cleaning
The dataset contains 947 rows and 8 columns. To ensure data integrity and consistency, we performed data validation and cleaning steps for each column:

- Handled missing values in the "calories," "carbohydrate," "sugar," and "protein" columns by filling them with the mean of their respective groups based on "category" and "servings".
- Unified the "Chicken Breast" category with the "Chicken" category to maintain consistency in categories.
- Unified extra values "4 as a snack" and "6 as a snack" in the "servings" column with "4" and "6," respectively, and changed the column type to integer.
- Replaced null values in the "high_traffic" column with "Low" to ensure all recipes have a traffic label.

## Exploratory Analysis
We conducted exploratory data analysis to gain insights into the data and answer specific questions:

- Explored two different types of graphics showing single variables, such as histograms and box plots.
- Created graphics showing two or more variables, like scatter plots and bar charts.
- Identified key findings, such as the most popular recipe categories and the correlation between recipe attributes and traffic.

## Model Development
We built two models to predict high traffic recipes:

- Fitted a baseline Logistic Regression model to establish a benchmark.
- Fitted a comparison Linear Support Vector Classification (SVC) model to evaluate its performance against the baseline.

## Model Evaluation
Based on evaluation metrics, the Logistic Regression model outperforms the Linear SVC model in predicting high traffic recipes. The Logistic Regression model achieves a Precision of 0.82, Recall of 0.80, and F1 Score of 0.81, while the Linear SVC model achieves a Precision of 0.80, Recall of 0.77, and F1 Score of 0.79.

## Metric for Monitoring
The primary business goal is to predict recipes with high traffic accurately. The chosen metric for monitoring is the accuracy of predictions for high traffic recipes. The Logistic Regression model achieves an accuracy of 76%, indicating a better-performing model compared to the Linear SVC model, which has an accuracy of 74%.

## Recommendations
Based on the analysis and model evaluation, we recommend the following actions to the business:

- Deploy the Logistic Regression model into production to predict high traffic recipes in real-time.
- Collect additional data, such as time to make, cost per serving, ingredients, and site duration time, to improve model performance.
- Implement feature engineering techniques to create more meaningful features and increase the number of categories.
- Monitor the accuracy of predictions regularly to ensure the model's performance remains consistent.
- Investigate further to identify other factors influencing recipe popularity and traffic.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Feel free to contribute to this project by creating pull requests or raising issues in the repository. Your contributions are valuable and appreciated!

## Get Started
To get started with the project, follow these steps:

1. Clone the repository: `git clone https://github.com/elfeenah/recipe-traffic-prediction.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the project: `python main.py`

## Project Team
This project was completed by:

- Nafisa Lawal Idris
<!-- Add other team members here -->
