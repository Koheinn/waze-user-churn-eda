# 🚗 Waze User Churn — Exploratory Data Analysis

> **Google Advanced Data Analytics Professional Certificate — Course 2 Portfolio Project**

An exploratory data analysis (EDA) project using synthetic Waze user data to investigate **user churn behavior**, identify patterns associated with churn, evaluate data quality, and communicate findings through Python visualizations and an executive summary.

---

## 📌 Project Overview

This project was completed as part of the **Google Advanced Data Analytics Professional Certificate**, specifically the Course 2 end-of-course project:

**Go Beyond the Numbers: Translate Data into Insights**

The objective was to perform exploratory data analysis on Waze user data as an initial step toward developing a future **machine learning model capable of predicting user churn**.

The analysis focuses on:

* Understanding the structure and quality of the dataset
* Identifying missing values and potential data-quality issues
* Examining distributions and outliers
* Exploring relationships between user behavior variables
* Investigating factors associated with churn
* Creating meaningful data visualizations
* Translating analytical findings into business insights
* Preparing an executive summary for stakeholders

The project follows Google's **PACE framework**:

**Plan → Analyze → Construct → Execute**

---

## 🎯 Business Problem

Waze wants to better understand why some users stop using the application.

The broader business objective is to eventually develop a machine learning model that can predict whether a user is likely to churn.

Before building such a model, the data needs to be thoroughly explored.

The key business questions explored in this project include:

1. What does the Waze user population look like?
2. What are the distributions of important user behavior variables?
3. Are there missing values or problematic observations?
4. Which variables appear to be associated with churn?
5. Does user activity differ between retained and churned users?
6. Is device type associated with churn?
7. Are there unusual relationships or anomalies that should be investigated?
8. What additional questions should be answered before building a churn prediction model?

---

## 📊 Dataset

The project uses a synthetic Waze dataset created specifically for the Google Advanced Data Analytics Professional Certificate in partnership with Waze.

### Dataset dimensions

* **Rows:** 14,999 users
* **Columns:** 13 in the original dataset
* **Target variable:** `label`
* **Target classes:** `retained` and `churned`

Each row represents one unique Waze user.

### Dataset Variables

| Variable                  | Type    | Description                                              |
| ------------------------- | ------- | -------------------------------------------------------- |
| `ID`                      | Integer | Unique identifier for each user                          |
| `label`                   | Object  | Whether the user was retained or churned                 |
| `sessions`                | Integer | Number of times the user opened the app during the month |
| `drives`                  | Integer | Number of driving occurrences of at least 1 km           |
| `device`                  | Object  | Device type used to start a session                      |
| `total_sessions`          | Float   | Estimated total sessions since onboarding                |
| `n_days_after_onboarding` | Integer | Number of days since user signup                         |
| `total_navigations_fav1`  | Integer | Total navigations to favorite place 1                    |
| `total_navigations_fav2`  | Integer | Total navigations to favorite place 2                    |
| `driven_km_drives`        | Float   | Total kilometers driven during the month                 |
| `duration_minutes_drives` | Float   | Total driving duration in minutes during the month       |
| `activity_days`           | Integer | Number of days the user opened the app                   |
| `driving_days`            | Integer | Number of days the user drove at least 1 km              |

---

## 🛠️ Technologies & Tools

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn
* Plotly

### Development Environment

* Jupyter Notebook

### Version Control

* Git
* GitHub

---

## 🧠 Methodology

The analysis follows the **PACE framework** used throughout the Google Advanced Data Analytics Professional Certificate.

### 1. Plan

The first stage focused on understanding the business problem and defining the analytical objectives.

The main objective was to investigate user behavior and identify variables that may be associated with churn.

Key questions included:

* Which variables are relevant to churn?
* What information needs to be cleaned?
* Which visualizations would best communicate the data?
* What potential issues should be investigated before modeling?

---

### 2. Analyze

The dataset was loaded into a Pandas DataFrame and examined using:

```python
df.head()
df.describe()
df.info()
df.shape
```

The initial inspection showed:

```text
14,999 rows
13 columns
```

There were **700 missing values in the `label` column**.

The other variables did not contain missing values.

The `ID` variable was identified as an identifier rather than a meaningful predictive feature and therefore should not be used as a modeling feature.

---

### 3. Construct

Several visualization techniques were used to understand the structure of the dataset.

These included:

* Box plots
* Histograms
* Scatter plots
* Pie charts
* Grouped histograms
* Filled histograms

The analysis also involved feature engineering, including:

```python
km_per_driving_day
```

and:

```python
percent_sessions_in_last_month
```

A further behavioral metric was created:

```python
monthly_drives_per_session_ratio
```

Outliers were also investigated and, for selected highly skewed variables, values above the 95th percentile were capped for exploratory purposes.

---

### 4. Execute

The final stage focused on translating the analysis into actionable insights.

The results were summarized through:

* Python visualizations
* Notebook conclusions
* An executive summary
* Recommendations for further investigation

---

# 🔍 Exploratory Data Analysis

## Dataset Structure

The original dataset contains:

```text
14,999 users
13 variables
```

The target variable is:

```text
label
```

with two possible values:

```text
retained
churned
```

There were **700 missing values in the churn label**, which should be investigated before using the data for supervised machine learning.

---

## 📈 Distribution of Sessions

The `sessions` variable is strongly **right-skewed**.

The median number of sessions was approximately:

```text
56 sessions
```

However, some users had more than 700 sessions.

This indicates that a relatively small number of users are extremely active compared with the majority of users.

---

## 🚗 Distribution of Drives

The `drives` variable has a similar right-skewed distribution.

The median number of drives was approximately:

```text
48 drives
```

while some users had more than 400 drives during the month.

This suggests that a small group of highly active users may have a significant influence on summary statistics.

---

## 📱 Device Distribution

The dataset contains both:

* iPhone users
* Android users

There are nearly twice as many iPhone users as Android users.

However, the proportion of churned versus retained users was relatively consistent across the two device categories.

Therefore, **device type did not appear to be a strong differentiator of churn in this EDA**.

---

## 📊 User Retention

The overall churn rate was approximately:

```text
17% churned
83% retained
```

In other words, fewer than one in five users in the dataset churned.

This class imbalance should be considered when developing a future churn prediction model.

---

# 📅 Activity Days vs. Driving Days

The analysis compared:

* `activity_days`
* `driving_days`

A useful business rule was identified:

> A user cannot have more driving days than activity days.

The scatter plot confirmed that none of the observations violated this relationship.

However, there was an interesting data-quality issue:

```text
Maximum activity_days = 31
Maximum driving_days  = 30
```

This raises the possibility that these variables were not collected over exactly the same time period.

This should be confirmed with the Waze data team before using these variables in a production model.

---

# 🛣️ Distance Driven and Churn

A new feature was created:

```python
km_per_driving_day = driven_km_drives / driving_days
```

Initially, division by zero produced infinite values for users with zero driving days.

These infinite values were converted to zero for the analysis.

The resulting distribution showed that some users had extremely high values.

The maximum observed value was approximately:

```text
15,420 km per driving day
```

which is physically implausible.

Therefore, users with values greater than approximately **1,200 km per driving day** were excluded from this particular visualization.

---

## 🚨 Key Churn Finding: Distance per Driving Day

One of the strongest observations from the EDA was that:

> **Users who traveled longer distances per driving day tended to have higher churn rates.**

The churn rate increased as mean kilometers driven per driving day increased.

This raises an important business question:

**Why are high-distance users more likely to churn?**

Potential explanations could include:

* Commercial or professional drivers
* Rideshare drivers
* Users with unusual driving patterns
* Navigation dissatisfaction during long journeys
* Different expectations among high-mileage users

These possibilities would need additional data before drawing a causal conclusion.

---

# 🚘 Driving Frequency and Churn

Another important finding was the relationship between `driving_days` and churn.

The analysis showed that:

> **Users who drove more frequently were generally less likely to churn.**

The churn rate was highest among users with very few or zero driving days.

Conversely, users who drove on many days during the month had substantially lower churn rates.

This produces an interesting contrast:

### Higher driving frequency

⬇️ Lower churn

### Higher distance per driving day

⬆️ Higher churn

This suggests that **frequency and intensity of use may capture different aspects of user behavior**.

Further investigation is needed to understand why these relationships move in opposite directions.

---

# 📆 Sessions in the Last Month

A new feature was created:

```python
percent_sessions_in_last_month = sessions / total_sessions
```

The median value was approximately:

```text
0.423
```

or:

```text
42.3%
```

This means that approximately half of the users had at least 42% of their total recorded sessions occur during the most recent month.

This was particularly interesting because the median user tenure was approximately:

```text
1,741 days
```

or nearly:

```text
4.8 years
```

The analysis therefore revealed an unusual pattern where many long-tenure users appeared to have a substantial proportion of their total sessions concentrated in the latest month.

This should be investigated with the Waze data team.

---

# ⚠️ Data Quality Findings

Several potential data-quality issues were identified during EDA.

### 1. Missing churn labels

There are:

```text
700 missing values
```

in the `label` column.

Before supervised learning, these records would need to be handled appropriately.

---

### 2. Extreme values

Several numerical variables contain strong right-skew and extreme observations.

Examples include:

* `sessions`
* `drives`
* `total_sessions`
* `driven_km_drives`
* `duration_minutes_drives`

The presence of these observations should be considered during future model development.

---

### 3. Physically implausible driving distances

Some users have extremely large values for:

```text
km_per_driving_day
```

with the maximum exceeding:

```text
15,000 km/day
```

These observations are unrealistic and require investigation.

---

### 4. Inconsistent monthly ranges

The maximum values were:

```text
activity_days = 31
driving_days  = 30
```

This could indicate different collection periods or another data-generation issue.

---

### 5. Unexpected recent activity

A large proportion of users appear to have a significant percentage of their lifetime sessions concentrated in the latest month.

The reason for this pattern is unclear and should be investigated.

---

# 📊 Outlier Analysis

Box plots revealed numerous potential outliers.

However, the presence of an outlier does not automatically mean the observation is incorrect.

For this dataset, many extreme observations appear to be caused by **right-skewed user behavior rather than obvious data-entry errors**.

For exploratory purposes, the following variables were capped at their 95th percentile:

* `sessions`
* `drives`
* `total_sessions`
* `driven_km_drives`
* `duration_minutes_drives`

The calculated thresholds were:

| Variable                  |  95th Percentile |
| ------------------------- | ---------------: |
| `sessions`                |              243 |
| `drives`                  |              201 |
| `total_sessions`          |           454.36 |
| `driven_km_drives`        |      8,889.79 km |
| `duration_minutes_drives` | 4,668.90 minutes |

This was done as an exploratory technique rather than as a definitive data-cleaning decision for a future production model.

---

# 📊 Visualizations

The project includes multiple visualizations designed to communicate different aspects of the data.

### Distribution Analysis

* Sessions box plot
* Sessions histogram
* Drives box plot
* Drives histogram
* Total sessions box plot
* Total sessions histogram
* Activity days histogram
* Driving days histogram
* Distance driven histogram
* Driving duration histogram

### Relationship Analysis

* Driving days vs. activity days scatter plot
* Retention by device
* Churn rate by kilometers driven per driving day
* Churn rate by number of driving days
* Percentage of sessions occurring in the last month

### Categorical Analysis

* Users by device
* Retained vs. churned users

---

# 💡 Key Insights

The most important findings from the EDA are:

### 1. Churn is relatively low

Approximately **17% of users churned**, while approximately **83% were retained**.

---

### 2. Device type does not appear strongly associated with churn

The churn-to-retention proportion was relatively consistent between iPhone and Android users.

---

### 3. Driving frequency is associated with lower churn

Users who drove on more days during the month tended to have lower churn rates.

---

### 4. Long-distance driving is associated with higher churn

Users with higher average kilometers driven per driving day tended to churn at higher rates.

---

### 5. User behavior variables are heavily skewed

Several usage variables have long right tails, indicating a relatively small population of extremely active users.

---

### 6. The dataset contains unusual activity patterns

A substantial number of long-tenure users appear to have a large percentage of their lifetime sessions concentrated in the latest month.

This requires further investigation.

---

### 7. Potential data-quality issues need to be resolved

Missing churn labels, implausible driving distances, and inconsistent monthly day counts should be investigated before developing a production-ready machine learning model.

---

# ❓ Questions for Further Investigation

The EDA generated several questions that should be explored in future analysis.

### Data quality

* Why are 700 churn labels missing?
* Were `activity_days` and `driving_days` collected over the same month?
* Why does `activity_days` have a maximum of 31 while `driving_days` has a maximum of 30?
* What caused the extremely high distance-per-driving-day values?

### User behavior

* Who are the users with extremely high numbers of drives?
* Are these professional or commercial drivers?
* Why do frequent drivers have lower churn?
* Why do high-distance-per-driving-day users have higher churn?

### User engagement

* Why do many long-tenure users have such a high percentage of sessions in the most recent month?
* Was there a specific event or product change that increased recent activity?
* Does engagement differ significantly by user tenure?

### Business strategy

* Which behavioral features are the strongest predictors of churn?
* Can high-risk users be identified early?
* What interventions could reduce churn?
* Can Waze create different retention strategies for different user segments?

---

# 🤖 Future Machine Learning Work

This EDA represents an important preparation stage for the broader churn prediction project.

The next stage could involve developing a supervised machine learning model to predict:

```text
P(user churns)
```

Potential future steps include:

1. Handle missing target labels.
2. Perform deeper data-quality investigation.
3. Engineer additional behavioral features.
4. Encode categorical variables.
5. Investigate feature correlations.
6. Split the data into training and testing sets.
7. Address class imbalance if necessary.
8. Train baseline classification models.
9. Compare multiple algorithms.
10. Evaluate precision, recall, F1-score, ROC-AUC, and other appropriate metrics.
11. Perform feature importance analysis.
12. Tune the best-performing model.
13. Translate model results into business recommendations.

Potential models could include:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

The final model should prioritize **business usefulness and interpretability**, rather than accuracy alone.

---

# 📁 Project Structure

A suggested repository structure is:

```text
waze-user-churn-eda/
│
├── Activity_Course_3_Waze_project_lab.ipynb
├── waze_dataset.csv
├── Waze-Course-2-executive-summary.pptx
├── README.md
└── .gitignore
```

### Files

| File                                       | Description                                                                                    |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `Activity_Course_3_Waze_project_lab.ipynb` | Complete Python notebook containing the EDA, feature engineering, analysis, and visualizations |
| `waze_dataset.csv`                         | Synthetic Waze user dataset used for the analysis                                              |
| `Waze-Course-2-executive-summary.pptx`     | Executive summary presenting the main analytical findings                                      |
| `README.md`                                | Project documentation                                                                          |

---

# 📓 Notebook Contents

The Jupyter Notebook contains four major sections:

### Part 1 — Imports & Data Loading

* Import Python libraries
* Load the Waze dataset
* Initialize the analysis environment

### Part 2 — Data Exploration

* Inspect dataset structure
* Review summary statistics
* Identify missing data
* Investigate potential outliers
* Examine numerical and categorical variables

### Part 3 — Data Visualization

* Create box plots
* Create histograms
* Create scatter plots
* Compare device types
* Analyze retention and churn
* Investigate driving behavior

### Part 4 — Evaluation & Results

* Perform additional EDA
* Engineer behavioral features
* Evaluate relationships with churn
* Identify data-quality concerns
* Summarize business insights
* Recommend areas for further investigation

---

# 📋 PACE Framework

This project demonstrates the four stages of Google's PACE framework.

| Stage         | Project Application                                     |
| ------------- | ------------------------------------------------------- |
| **Plan**      | Define the churn problem and analytical questions       |
| **Analyze**   | Inspect, clean, summarize, and explore the dataset      |
| **Construct** | Build visualizations and engineer useful features       |
| **Execute**   | Communicate insights and identify next analytical steps |

---

# 🧰 Skills Demonstrated

This project demonstrates practical experience with:

### Python

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly

### Data Analysis

* Exploratory Data Analysis
* Data cleaning
* Missing-value analysis
* Descriptive statistics
* Distribution analysis
* Outlier detection
* Feature engineering
* Data validation

### Data Visualization

* Histograms
* Box plots
* Scatter plots
* Pie charts
* Grouped visualizations
* Churn-rate visualizations

### Business Analytics

* Translating data into insights
* Identifying business questions
* Communicating analytical findings
* Executive-level reporting
* Connecting technical findings to business decisions

### Data Science

* Churn analysis
* Behavioral feature engineering
* Preparing data for machine learning
* Identifying candidate predictive features
* Evaluating data quality before modeling

---

# 🎓 Certificate

This project was completed as part of the:

**Google Advanced Data Analytics Professional Certificate**

Course:

**Course 2 — Go Beyond the Numbers: Translate Data into Insights**

Project:

**Waze User Churn — Exploratory Data Analysis**

The project applies concepts learned throughout the course, including:

* Python programming
* Pandas
* NumPy
* Data visualization
* Exploratory data analysis
* Data cleaning
* Feature engineering
* Business communication
* The PACE problem-solving framework

---

# 📌 Important Note

The Waze dataset used in this project is **synthetic data created for the Google Advanced Data Analytics Professional Certificate**.

The analysis is therefore intended for **educational and portfolio purposes** and should not be interpreted as analysis of actual Waze customer data.

---

# 👤 Author

**Heinn Htet Zan**

Computer Science Student | Full-Stack Developer | Aspiring Data Scientist / Machine Learning Engineer

### Connect with me

* **GitHub:** https://github.com/Koheinn
* **LinkedIn:** https://www.linkedin.com/in/heinn-htet-zan-040794291/
* **Portfolio:** https://hhzportfolio.netlify.app

---

# ⭐ Project Highlights

> **Business Problem:** Understand and investigate Waze user churn.

> **Dataset:** 14,999 synthetic Waze users.

> **EDA Finding:** Approximately 17% of users churned.

> **Key Behavioral Finding:** More frequent driving was associated with lower churn, while higher average distance per driving day was associated with higher churn.

> **Data Quality Finding:** Missing churn labels, extreme driving values, and inconsistent monthly day counts require further investigation.

> **Next Step:** Use the insights from EDA to develop and evaluate a machine learning model for predicting user churn.

---

## 🚀 What This Project Demonstrates

This project goes beyond simply creating charts. It demonstrates the complete analytical workflow of:

```text
Business Problem
       ↓
Data Understanding
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Visualization
       ↓
Feature Engineering
       ↓
Insight Discovery
       ↓
Business Questions
       ↓
Future Machine Learning
```

The ultimate goal is to transform raw user activity data into **actionable insights that can support Waze's user-retention strategy**.
