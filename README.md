# AI-Sales-Forecasting-Explanation-Engine
An AI-driven Sales Forecasting & Explainability Engine. Using Machine Learning techniques along with Historical Sales Data and SHAP explainability we can identify trends in revenue, understand what factors are driving our predictions and turn sales forecasting into useful insights using an interactive Streamlit dashboard.

Dataset Overview - 
9,245 order records. 
15,608 order-item records. 
90 product records. 
Customer information and product attributes.

Approach - Project Workflow
1) Data Cleaning & Integration (Scripts 01–06): Inspected and cleaned four source datasets, standardized dates, handled missing values and duplicates, and created a consolidated daily sales dataset.
2) Exploratory Data Analysis (Script 07): Analyzed historical revenue trends, day-of-week patterns, and monthly seasonality.
3) Feature Engineering (Script 08): Created calendar features, historical revenue lags (1, 7, and 14 days), and rolling statistics (7-day average, 7-day standard deviation, and 30-day average).
4) Model Development (Script 09b): Developed a Random Forest regression model using chronological training, validation, and test splits. Hyperparameters were selected using validation performance, and the final model was evaluated against a weekly-naive baseline.
5) Revenue Forecasting (Script 10): Generated 30-day daily revenue forecasts using recursive predictions, where previous predictions become inputs for subsequent forecast steps.
6) Explainability (Script 11): Integrated SHAP (SHapley Additive exPlanations) to quantify feature contributions and explain individual revenue predictions.
7) Dashboard (Script 12): Developed an interactive Streamlit dashboard to explore historical sales, predictions, revenue forecasts, model performance, and SHAP explanations.

Key Business Insights -
1) Seasonality: November and December contribute approximately 22% of annual revenue. November average daily revenue was ₹3,842, around 56% higher than the average for all other months. Higher order volume drives November peaks.
2) Business Growth: Revenue increased 21.1% year over year, driven mainly by an 18.7% increase in orders.
3) Discount Effectiveness: Discounted orders have a 22% lower average order value than non-discounted orders.
4) Returns & Cancellations: Approximately 9.3% of gross item revenue is associated with returned and cancelled orders.
5) Sales Patterns: Weekends generate approximately 10.8% more revenue than weekdays, while Furniture is the largest revenue-contributing category.

Explainability
SHAP feature contributions help identify the factors influencing individual revenue predictions. Combined with historical business insights, the explainability layer makes model behaviour more transparent and helps users interpret forecast drivers.

Tech Stack -
•	Programming: Python
•	Data Processing: Pandas
•	Machine Learning: Scikit-learn, Random Forest Regression
•	Explainable AI: SHAP
•	Visualization: Matplotlib
•	Dashboard: Streamlit

Usage - 
streamlit run 12_dashboard.py

Conclusion - 
This project explored the store's sales performance, identified revenue trends, and developed a 30-day revenue forecast. The analysis provided a better understanding of sales patterns and the factors influencing revenue.
