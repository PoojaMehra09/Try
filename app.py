import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Hide Streamlit warnings
st.title('Hotel Booking Analytics Dashboard')

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv('hotel_bookings.csv')
    return df

uploaded_file = st.sidebar.file_uploader("Upload your hotel_bookings.csv file", type=['csv'])
df = load_data(uploaded_file)

# Preprocessing for Classification/Clustering/Regression
def preprocess(df):
    df_proc = df.copy()
    # Fill NA and encode categories
    for col in df_proc.select_dtypes('object').columns:
        df_proc[col] = df_proc[col].fillna(df_proc[col].mode()[0])
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col].astype(str))
    df_proc = df_proc.fillna(df_proc.median(numeric_only=True))
    return df_proc

df_proc = preprocess(df)

# ---------------- TAB SETUP ---------------- #
tabs = st.tabs([
    "Data Visualization", "Classification", "Clustering", 
    "Association Rule Mining", "Regression"
])

# ---------------- TAB 1: DATA VISUALIZATION ---------------- #
with tabs[0]:
    st.header("1. Data Visualization")
    st.write("Explore descriptive, actionable insights on hotel bookings data. All charts below are interactive and support filters.")

    # Example filters
    col1, col2 = st.columns(2)
    hotel_types = df['hotel'].unique()
    hotel_choice = col1.selectbox("Hotel Type", ["All"] + list(hotel_types))
    month_choice = col2.selectbox("Arrival Month", ["All"] + list(df['arrival_date_month'].unique()))

    filter_df = df.copy()
    if hotel_choice != "All":
        filter_df = filter_df[filter_df['hotel'] == hotel_choice]
    if month_choice != "All":
        filter_df = filter_df[filter_df['arrival_date_month'] == month_choice]

    # 1. Booking Trends over Months
    st.subheader("Booking Trends per Month")
    monthly_counts = filter_df.groupby('arrival_date_month').size().sort_index()
    st.bar_chart(monthly_counts)
    st.caption("Shows seasonality in bookings—peak months are visible.")

    # 2. Average Daily Rate by Hotel Type
    st.subheader("Average Daily Rate by Hotel Type")
    adr_by_hotel = filter_df.groupby('hotel')['adr'].mean()
    st.bar_chart(adr_by_hotel)
    st.caption("Compares price points of City vs Resort hotels.")

    # 3. Distribution of Stays in Weekend Nights
    st.subheader("Weekend Nights Stay Distribution")
    fig, ax = plt.subplots()
    ax.hist(filter_df['stays_in_weekend_nights'], bins=20)
    ax.set_xlabel('Stays in Weekend Nights')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    st.caption("Most bookings are for short weekend stays.")

    # 4. Booking Cancellations by Lead Time
    st.subheader("Cancellations vs. Lead Time")
    fig, ax = plt.subplots()
    sns.boxplot(data=filter_df, x='is_canceled', y='lead_time', ax=ax)
    st.pyplot(fig)
    st.caption("Bookings with longer lead time tend to have higher cancellations.")

    # 5. Countrywise Booking Frequency
    st.subheader("Top Countries by Booking Volume")
    country_counts = filter_df['country'].value_counts().head(10)
    st.bar_chart(country_counts)
    st.caption("Shows countries that bring most bookings.")

    # 6. Average Guests per Booking
    st.subheader("Avg. Total Guests per Booking")
    filter_df['total_guests'] = filter_df['adults'] + filter_df['children'].fillna(0) + filter_df['babies']
    avg_guests = filter_df['total_guests'].mean()
    st.metric("Average Guests", f"{avg_guests:.2f}")

    # 7. Room Type vs. Price
    st.subheader("Room Type vs. ADR")
    fig, ax = plt.subplots()
    sns.boxplot(data=filter_df, x='reserved_room_type', y='adr', ax=ax)
    st.pyplot(fig)
    st.caption("Shows price segmentation by room type.")

    # 8. Market Segment Impact on Cancellations
    st.subheader("Cancellations by Market Segment")
    cancels_by_market = filter_df.groupby('market_segment')['is_canceled'].mean()
    st.bar_chart(cancels_by_market)
    st.caption("Segments with highest cancellation ratios are visible.")

    # 9. Repeat Guests Trend
    st.subheader("Trend of Repeat Guests")
    repeat_rate = filter_df['is_repeated_guest'].mean() * 100
    st.metric("Repeat Guest Rate (%)", f"{repeat_rate:.2f}")
    
    # 10. Distribution of Children per Booking
    st.subheader("Distribution of Children")
    fig, ax = plt.subplots()
    ax.hist(filter_df['children'].fillna(0), bins=10)
    ax.set_xlabel('Number of Children')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    st.caption("Most bookings have no children, a few with multiple.")

    # Download current filtered data
    st.download_button("Download filtered data", data=filter_df.to_csv(index=False), file_name="filtered_data.csv")

# ---------------- TAB 2: CLASSIFICATION ---------------- #
with tabs[1]:
    st.header("2. Classification: Predict Booking Cancellation")
    st.write("Compare multiple classifiers and predict if a booking will be canceled.")

    # Feature and target setup
    features = [col for col in df_proc.columns if col != 'is_canceled']
    X = df_proc[features]
    y = df_proc['is_canceled']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    results = []
    preds = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds[name] = y_pred
        results.append({
            'Model': name,
            'Train Acc': accuracy_score(y_train, model.predict(X_train)),
            'Test Acc': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred)
        })
    result_df = pd.DataFrame(results)
    st.dataframe(result_df.round(3))
    st.caption("Table shows performance of all classification algorithms.")

    # Dropdown for confusion matrix
    conf_model = st.selectbox("Select model for Confusion Matrix", list(models.keys()))
    cm = confusion_matrix(y_test, preds[conf_model])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Canceled', 'Canceled'], 
                yticklabels=['Not Canceled', 'Canceled'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

    # ROC Curve for all models
    st.subheader("ROC Curves")
    fig, ax = plt.subplots()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # Upload new data for prediction
    st.subheader("Upload Data for Cancellation Prediction")
    upload_pred = st.file_uploader("Upload CSV (no is_canceled column)", type=['csv'], key="pred")
    if upload_pred:
        df_new = pd.read_csv(upload_pred)
        df_new_proc = preprocess(df_new)
        pred_model = st.selectbox("Model for Prediction", list(models.keys()), key="pred2")
        chosen_model = models[pred_model]
        pred_result = chosen_model.predict(df_new_proc[features])
        result_df = df_new.copy()
        result_df['predicted_is_canceled'] = pred_result
        st.dataframe(result_df)
        csv_out = result_df.to_csv(index=False)
        st.download_button("Download Predictions", csv_out, "predictions.csv")

# ---------------- TAB 3: CLUSTERING ---------------- #
with tabs[2]:
    st.header("3. Clustering: Customer Segmentation")
    st.write("Segment customers based on booking behaviors using K-Means clustering.")

    # Features for clustering
    clustering_features = ['lead_time', 'adults', 'children', 'babies', 'adr', 'stays_in_weekend_nights', 'stays_in_week_nights']
    cluster_df = df_proc[clustering_features]
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_df)

    # Elbow method chart
    st.subheader("Elbow Method for K selection")
    k_range = range(2, 11)
    inertia = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(cluster_scaled)
        inertia.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(k_range, inertia, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    st.pyplot(fig)

    # Cluster slider
    n_clusters = st.slider("Select number of clusters", 2, 10, 3)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = km.fit_predict(cluster_scaled)
    cluster_df['cluster'] = clusters

    # Customer persona table
    persona = cluster_df.groupby('cluster').mean().round(2)
    st.dataframe(persona)
    st.caption("Table shows average characteristics for each cluster.")

    # Download full data with clusters
    df_with_cluster = df.copy()
    df_with_cluster['cluster'] = clusters
    st.download_button("Download Data with Cluster Labels", df_with_cluster.to_csv(index=False), "clustered_data.csv")

# ---------------- TAB 4: ASSOCIATION RULE MINING ---------------- #
with tabs[3]:
    st.header("4. Association Rule Mining")
    st.write("Find interesting booking patterns using Apriori.")

    # For simplicity, choose 2 categorical columns
    apriori_cols = st.multiselect("Select 2+ columns for Apriori", df.select_dtypes('object').columns, default=['hotel', 'meal'])
    min_support = st.slider("Minimum Support", 0.01, 0.5, 0.1, 0.01)
    min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05)
    if len(apriori_cols) >= 2:
        # Prepare data in transaction format
        df_ap = df[apriori_cols].astype(str)
        one_hot = pd.get_dummies(df_ap)
        freq_items = apriori(one_hot, min_support=min_support, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
        rules = rules.sort_values('confidence', ascending=False).head(10)
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        st.caption("Top 10 association rules by confidence.")

# ---------------- TAB 5: REGRESSION ---------------- #
with tabs[4]:
    st.header("5. Regression Insights")
    st.write("Forecast monthly bookings, explore price drivers, and compare regression models.")

    # Prepare regression targets and features
    # Example: Forecast bookings per month
    st.subheader("Forecast Number of Bookings per Month")
    reg_df = df.copy()
    month_map = {m: i+1 for i, m in enumerate(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])}
    reg_df['month_num'] = reg_df['arrival_date_month'].map(month_map)
    bookings_per_month = reg_df.groupby('month_num').size().reset_index(name='bookings')
    X = bookings_per_month[['month_num']]
    y = bookings_per_month['bookings']

    # Linear regression
    reg = LinearRegression()
    reg.fit(X, y)
    pred = reg.predict(X)
    fig, ax = plt.subplots()
    ax.scatter(X, y, label='Actual')
    ax.plot(X, pred, color='red', label='Predicted')
    ax.set_xlabel('Month')
    ax.set_ylabel('Bookings')
    ax.set_title('Monthly Bookings Forecast (Linear Regression)')
    ax.legend()
    st.pyplot(fig)
    st.caption("Linear regression on monthly booking counts. Shows seasonality trend.")

    # ADR prediction using Ridge, Lasso, DT
    st.subheader("Predict ADR (Price) Using Different Regressors")
    reg_X = df_proc.drop(columns=['adr', 'is_canceled'])
    reg_y = df_proc['adr']

    X_train, X_test, y_train, y_test = train_test_split(reg_X, reg_y, test_size=0.2, random_state=42)
    regressors = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42)
    }
    reg_results = []
    for name, model in regressors.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = model.score(X_test, y_test)
        rmse = np.sqrt(((y_test - y_pred)**2).mean())
        reg_results.append({'Regressor': name, 'R2': r2, 'RMSE': rmse})
    st.dataframe(pd.DataFrame(reg_results).round(3))
    st.caption("Model comparison for predicting price (ADR).")

    # Feature importances for Decision Tree
    dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    imp_df = pd.DataFrame({
        'Feature': reg_X.columns,
        'Importance': dt.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    st.bar_chart(imp_df.set_index('Feature'))
    st.caption("Top 10 features influencing room price (ADR) by Decision Tree.")

# ---------------- END OF APP ---------------- #

