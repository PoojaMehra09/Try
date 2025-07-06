import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk

st.set_page_config(page_title="Hotel Bookings Analytics Dashboard", layout="wide")

# ----------------- LOAD DATA ------------------- #
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv('hotel_bookings.csv')
    return df

uploaded_file = st.sidebar.file_uploader("Upload your hotel_bookings.csv file", type=['csv'], help="Upload the hotel bookings dataset as CSV.")
df = load_data(uploaded_file)

# ----------------- PREPROCESSING ------------------- #
def preprocess(df):
    df_proc = df.copy()
    for col in df_proc.select_dtypes('object').columns:
        df_proc[col] = df_proc[col].fillna(df_proc[col].mode()[0])
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col].astype(str))
    df_proc = df_proc.fillna(df_proc.median(numeric_only=True))
    return df_proc

df_proc = preprocess(df)

# ----------------- COUNTRY-TO-LAT/LON DICT (for map) ------------------- #
country_lat_lon = {
    'PRT': (39.3999, -8.2245),
    'GBR': (55.3781, -3.4360),
    'FRA': (46.6034, 1.8883),
    'ESP': (40.4637, -3.7492),
    'ITA': (41.8719, 12.5674),
    'IRL': (53.1424, -7.6921),
    'DEU': (51.1657, 10.4515),
    'BEL': (50.5039, 4.4699),
    'NLD': (52.1326, 5.2913),
    'BRA': (-14.2350, -51.9253),
    # Add more as needed
}

# ----------------- TABS ------------------- #
tabs = st.tabs([
    "Data Visualization", "Classification", "Clustering",
    "Association Rule Mining", "Regression"
])

# ----------------- TAB 1: DATA VISUALIZATION ------------------- #
with tabs[0]:
    st.header("1. Data Visualization")
    st.markdown("Explore descriptive, actionable insights on hotel bookings data.")

    # --- Preview/Summary ---
    with st.expander("Show data preview and summary"):
        st.write("First 5 rows of dataset:")
        st.dataframe(df.head())
        st.write("Numerical summary:")
        st.dataframe(df.describe())

    # --- Filters with Tooltips ---
    col1, col2 = st.columns(2)
    hotel_types = df['hotel'].unique()
    hotel_choice = col1.selectbox("Hotel Type", ["All"] + list(hotel_types),
                                  help="Filter by hotel type.")
    month_choice = col2.selectbox("Arrival Month", ["All"] + list(df['arrival_date_month'].unique()),
                                  help="Filter by arrival month.")

    filter_df = df.copy()
    if hotel_choice != "All":
        filter_df = filter_df[filter_df['hotel'] == hotel_choice]
    if month_choice != "All":
        filter_df = filter_df[filter_df['arrival_date_month'] == month_choice]

    # --- Color Pickers ---
    col_bar, col_line = st.columns(2)
    color1 = col_bar.color_picker("Pick color for Bar/Histograms", "#1f77b4", key="color1")
    color2 = col_line.color_picker("Pick color for Line/Pareto", "#d62728", key="color2")

    # 1. Booking Trends over Months
    st.subheader("Booking Trends per Month")
    monthly_counts = filter_df.groupby('arrival_date_month').size().sort_index()
    fig, ax = plt.subplots()
    ax.bar(monthly_counts.index, monthly_counts.values, color=color1)
    ax.set_xlabel('Month')
    ax.set_ylabel('Bookings')
    st.pyplot(fig)
    st.caption("Shows seasonality in bookings—peak months are visible.")

    # 2. Average Daily Rate by Hotel Type
    st.subheader("Average Daily Rate by Hotel Type")
    adr_by_hotel = filter_df.groupby('hotel')['adr'].mean()
    fig, ax = plt.subplots()
    ax.bar(adr_by_hotel.index, adr_by_hotel.values, color=color1)
    ax.set_xlabel("Hotel Type")
    ax.set_ylabel("Average Daily Rate (ADR)")
    st.pyplot(fig)

    # 3. Distribution of Stays in Weekend Nights
    st.subheader("Weekend Nights Stay Distribution")
    fig, ax = plt.subplots()
    ax.hist(filter_df['stays_in_weekend_nights'], bins=20, color=color1)
    ax.set_xlabel('Stays in Weekend Nights')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # 4. Booking Cancellations by Lead Time
    st.subheader("Cancellations vs. Lead Time")
    fig, ax = plt.subplots()
    sns.boxplot(data=filter_df, x='is_canceled', y='lead_time', ax=ax)
    st.pyplot(fig)

    # 5. Countrywise Booking Frequency
    st.subheader("Top Countries by Booking Volume")
    country_counts = filter_df['country'].value_counts().head(10)
    fig, ax = plt.subplots()
    ax.bar(country_counts.index, country_counts.values, color=color1)
    ax.set_xlabel("Country")
    ax.set_ylabel("Number of Bookings")
    st.pyplot(fig)

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

    # 8. Market Segment Impact on Cancellations
    st.subheader("Cancellations by Market Segment")
    cancels_by_market = filter_df.groupby('market_segment')['is_canceled'].mean()
    fig, ax = plt.subplots()
    ax.bar(cancels_by_market.index, cancels_by_market.values, color=color1)
    ax.set_xlabel("Market Segment")
    ax.set_ylabel("Cancellation Ratio")
    st.pyplot(fig)

    # 9. Repeat Guests Trend
    st.subheader("Trend of Repeat Guests")
    repeat_rate = filter_df['is_repeated_guest'].mean() * 100
    st.metric("Repeat Guest Rate (%)", f"{repeat_rate:.2f}")

    # 10. Distribution of Children per Booking
    st.subheader("Distribution of Children")
    fig, ax = plt.subplots()
    ax.hist(filter_df['children'].fillna(0), bins=10, color=color1)
    ax.set_xlabel('Number of Children')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # --- Geographical Visualization (Pydeck) ---
    st.subheader("Geographical Distribution of Bookings")
    map_df = df['country'].value_counts().reset_index()
    map_df.columns = ['country', 'count']
    map_df['lat'] = map_df['country'].map(lambda x: country_lat_lon.get(x, (0,0))[0])
    map_df['lon'] = map_df['country'].map(lambda x: country_lat_lon.get(x, (0,0))[1])
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=map_df[map_df['lat'] != 0],
                get_position='[lon, lat]',
                get_radius=40000,
                get_fill_color='[200, 30, 0, 160]',
                pickable=True,
            ),
        ],
    ))
    st.caption("Bubble size shows volume of bookings from each country.")

    # --- Correlation Matrix Heatmap ---
    st.subheader("Correlation Matrix (Heatmap)")
    corr = df_proc.corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.caption("Shows relationships between numeric variables in the data.")

    # --- Pareto Chart for Cancellations ---
    st.subheader("Pareto Chart: Cancellations by Country")
    canc_pareto = df[df['is_canceled']==1]['country'].value_counts().head(10)
    cum_pct = canc_pareto.cumsum()/canc_pareto.sum()
    fig, ax1 = plt.subplots()
    ax1.bar(canc_pareto.index, canc_pareto.values, color=color1)
    ax2 = ax1.twinx()
    ax2.plot(canc_pareto.index, cum_pct.values, color=color2, marker="D", ms=7)
    ax2.axhline(0.8, color="C2", linestyle="dashed")
    ax1.set_ylabel('Cancellations')
    ax2.set_ylabel('Cumulative %')
    st.pyplot(fig)
    st.caption("80% of cancellations come from these top countries (Pareto Principle).")

    # --- Download current filtered data
    st.download_button("Download filtered data", data=filter_df.to_csv(index=False), file_name="filtered_data.csv",
        help="Download the currently filtered data as CSV.")

# ----------------- TAB 2: CLASSIFICATION ------------------- #
with tabs[1]:
    st.header("2. Classification: Predict Booking Cancellation")
    st.markdown("Compare multiple classifiers and predict if a booking will be canceled.")

    # --- Data Preview/Summary
    with st.expander("Show data preview and summary"):
        st.write("First 5 rows of data for classification:")
        st.dataframe(df.head())
        st.write("Data summary (numerical columns):")
        st.dataframe(df.describe())

    # --- Feature and target setup
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

    with st.spinner('Training classification models...'):
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
    st.success('All models trained!')

    result_df = pd.DataFrame(results)
    st.dataframe(result_df.round(3))
    st.caption("Table shows performance of all classification algorithms.")

    # Dropdown for confusion matrix
    conf_model = st.selectbox("Select model for Confusion Matrix", list(models.keys()),
                              help="Pick a model to view its confusion matrix and classification details.")
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
    color_roc = st.color_picker("Pick ROC Curve color base", "#1f77b4", key="roc_color",
                                help="Select base color for ROC curves.")
    fig, ax = plt.subplots()
    for idx, (name, model) in enumerate(models.items()):
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})", color=f"C{idx}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # Upload new data for prediction
    st.subheader("Upload Data for Cancellation Prediction")
    upload_pred = st.file_uploader("Upload CSV (no is_canceled column)", type=['csv'], key="pred",
                                   help="Upload new booking data (excluding the 'is_canceled' column) for predictions.")
    if upload_pred:
        df_new = pd.read_csv(upload_pred)
        df_new_proc = preprocess(df_new)
        pred_model = st.selectbox("Model for Prediction", list(models.keys()), key="pred2",
                                  help="Choose the classifier model to use for prediction.")
        with st.spinner('Generating predictions...'):
            chosen_model = models[pred_model]
            pred_result = chosen_model.predict(df_new_proc[features])
            result_df = df_new.copy()
            result_df['predicted_is_canceled'] = pred_result
            st.dataframe(result_df)
            csv_out = result_df.to_csv(index=False)
            st.download_button("Download Predictions", csv_out, "predictions.csv",
                               help="Download the predictions as a CSV file.")

# ----------------- TAB 3: CLUSTERING ------------------- #
with tabs[2]:
    st.header("3. Clustering: Customer Segmentation")
    st.markdown("Segment customers based on booking behaviors using K-Means clustering.")

    with st.expander("Show data preview and summary"):
        st.write("First 5 rows for clustering:")
        st.dataframe(df.head())
        st.write("Numerical summary:")
        st.dataframe(df.describe())

    clustering_features = ['lead_time', 'adults', 'children', 'babies', 'adr', 'stays_in_weekend_nights', 'stays_in_week_nights']
    cluster_df = df_proc[clustering_features]
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_df)

    # Elbow method with progress bar
    st.subheader("Elbow Method for K selection")
    k_range = range(2, 11)
    inertia = []
    progress = st.progress(0, text="Calculating elbow curve...")
    for i, k in enumerate(k_range):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(cluster_scaled)
        inertia.append(km.inertia_)
        progress.progress((i+1)/len(k_range), text=f"Calculating for k={k}...")
    progress.empty()
    fig, ax = plt.subplots()
    ax.plot(list(k_range), inertia, marker='o', color=color1)
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    st.pyplot(fig)

    # Cluster slider with tooltip
    n_clusters = st.slider("Select number of clusters", 2, 10, 3,
                           help="Choose how many customer segments to create. The Elbow chart can help you decide.")
    km = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = km.fit_predict(cluster_scaled)
    cluster_df['cluster'] = clusters

    # Persona Table
    persona = cluster_df.groupby('cluster').mean().round(2)
    st.dataframe(persona)
    st.caption("Table shows average characteristics for each cluster.")

    # Download data with clusters
    df_with_cluster = df.copy()
    df_with_cluster['cluster'] = clusters
    st.download_button("Download Data with Cluster Labels", df_with_cluster.to_csv(index=False), "clustered_data.csv",
        help="Download the full dataset with assigned cluster labels.")

    # --- Personalized Recommendations ---
    st.subheader("Personalized Recommendations by Cluster")
    for i, row in persona.iterrows():
        msg = f"**Cluster {i}:** "
        if row['lead_time'] > 100:
            msg += "Consider targeting these customers with early-bird offers. "
        if row['stays_in_weekend_nights'] > 2:
            msg += "Promote weekend packages. "
        if row['adults'] >= 2:
            msg += "Family/group offers may work well. "
        if row['adr'] > persona['adr'].mean():
            msg += "Upsell premium services or room upgrades. "
        st.markdown(msg)

# ----------------- TAB 4: ASSOCIATION RULE MINING ------------------- #
with tabs[3]:
    st.header("4. Association Rule Mining")
    st.markdown("Find interesting booking patterns using Apriori.")

    with st.expander("Show data preview and summary"):
        st.write("First 5 rows for ARM:")
        st.dataframe
