# Kaggle Challenge: Churn prediction 25/26

Names:

- Akilan THEVARAJA
- Romain ETIENNE

## Rules

**Objective**: Predict if used (`user_id`) will churn within the next 10 days ($\Leftrightarrow \text{after 2018-11-20}$).

Churns $\Leftrightarrow$ X visits `Cancellation Confirmation` page.

## Steps

### Dataset Samples

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>status</th>
      <th>gender</th>
      <th>firstName</th>
      <th>level</th>
      <th>lastName</th>
      <th>userId</th>
      <th>ts</th>
      <th>auth</th>
      <th>page</th>
      <th>sessionId</th>
      <th>location</th>
      <th>itemInSession</th>
      <th>userAgent</th>
      <th>method</th>
      <th>length</th>
      <th>song</th>
      <th>artist</th>
      <th>time</th>
      <th>registration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8415245</th>
      <td>200</td>
      <td>M</td>
      <td>August</td>
      <td>paid</td>
      <td>Clark</td>
      <td>1207487</td>
      <td>1540518498000</td>
      <td>Logged In</td>
      <td>NextSong</td>
      <td>112798</td>
      <td>Riverside-San Bernardino-Ontario, CA</td>
      <td>400</td>
      <td>Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:1...</td>
      <td>PUT</td>
      <td>236.04200</td>
      <td>Hostile [Live Version]</td>
      <td>Pantera</td>
      <td>2018-10-26 01:48:18</td>
      <td>2018-09-10 17:03:47</td>
    </tr>
    <tr>
      <th>12828473</th>
      <td>200</td>
      <td>F</td>
      <td>Alani</td>
      <td>paid</td>
      <td>Kane</td>
      <td>1325330</td>
      <td>1541672419000</td>
      <td>Logged In</td>
      <td>NextSong</td>
      <td>159518</td>
      <td>Tampa-St. Petersburg-Clearwater, FL</td>
      <td>74</td>
      <td>Mozilla/5.0 (Windows NT 6.1; rv:31.0) Gecko/20...</td>
      <td>PUT</td>
      <td>194.03710</td>
      <td>Karibien</td>
      <td>Air France</td>
      <td>2018-11-08 10:20:19</td>
      <td>2018-09-05 06:02:19</td>
    </tr>
    <tr>
      <th>13490632</th>
      <td>200</td>
      <td>F</td>
      <td>Angelina</td>
      <td>paid</td>
      <td>Singh</td>
      <td>1495032</td>
      <td>1541826321000</td>
      <td>Logged In</td>
      <td>NextSong</td>
      <td>174513</td>
      <td>Santa Maria-Santa Barbara, CA</td>
      <td>216</td>
      <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like...</td>
      <td>PUT</td>
      <td>278.07302</td>
      <td>Through The Wire</td>
      <td>Kanye West</td>
      <td>2018-11-10 05:05:21</td>
      <td>2018-08-24 13:24:27</td>
    </tr>
    <tr>
      <th>9329450</th>
      <td>200</td>
      <td>M</td>
      <td>Nathaniel</td>
      <td>paid</td>
      <td>Norris</td>
      <td>1525270</td>
      <td>1540794304000</td>
      <td>Logged In</td>
      <td>NextSong</td>
      <td>128771</td>
      <td>San Francisco-Oakland-Hayward, CA</td>
      <td>120</td>
      <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4...</td>
      <td>PUT</td>
      <td>132.85832</td>
      <td>Ride A White Swan</td>
      <td>T. Rex</td>
      <td>2018-10-29 06:25:04</td>
      <td>2018-06-23 13:42:06</td>
    </tr>
    <tr>
      <th>13901146</th>
      <td>200</td>
      <td>M</td>
      <td>Landyn</td>
      <td>paid</td>
      <td>Evans</td>
      <td>1849720</td>
      <td>1541985370000</td>
      <td>Logged In</td>
      <td>NextSong</td>
      <td>179047</td>
      <td>Atlanta-Sandy Springs-Roswell, GA</td>
      <td>74</td>
      <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4...</td>
      <td>PUT</td>
      <td>266.65751</td>
      <td>Black Planet (Remastered)</td>
      <td>Sisters Of Mercy</td>
      <td>2018-11-12 01:16:10</td>
      <td>2018-08-25 21:44:42</td>
    </tr>
  </tbody>
</table>
</div>

### Cleaning & Analysis

Perform the data cleaning:

- **Type Casting**: Ensure `userId` is a string and `ts`/`registration` are datetime objects.
- **Missing Values**: Analyze and handle `NaN` values (especially in `userId`).
- **Define Churn**: Create a binary target variable `churn` (1 if user visited `Cancellation Confirmation` in the next 10 days).
- **EDA**:
  - Plot distribution of Churn (Check for **Class Imbalance**).
  - Compare behavior: "Average songs played per session for Churners vs. Non-Churners".
  - Visualization of standard users journey to `churn`.

### Feature Engineering

**Crucial Step**: Transform "Event-Level Data" (logs) into "User-Level Data" (one row per user).

- **User-Level Aggregations**:
  - _Activity_: Total songs, thumbs up/down, errors.
  - _Engagement_: Avg session duration, avg songs per session.
  - _Temporal_: Days since registration, time since last active session.
- **Encoding**: Gender, Level, Location.

### Preprocessing Pipeline

Use pipelines to:

- **Train/Test Split**: Use Stratified Split to maintain churn rate.
- **Handle Imbalance**: Use `SMOTE` or `class_weight='balanced'`.
- **Scale**: Apply `StandardScaler` or `MinMaxScaler` for numerical features.

### Modeling & Evaluation

- **Model Selection**:
  - Baseline (Dummy Classifier).
  - Logistic Regression, Random Forest.
  - **Gradient Boosting**: XGBoost, LightGBM, or CatBoost.
- **Hyperparameter Tuning**: GridSearch or RandomSearch.
- **Metrics**:
  - **F1-Score** & **Recall** (Priority over accuracy).
  - **ROC-AUC**.

### Interpretation & Explainability

- **Feature Importance**: Identify top drivers of churn.
- **SHAP Values**: Explain individual predictions (why did _this_ specific user churn?).

### Final evaluations and submissions

Generate predictions on the test set and submit.
