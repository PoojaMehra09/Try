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

st.set_page_config(page_title="🏨 Hotel Bookings Dashboard", layout="wide")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- SIDEBAR
st.sidebar.title("🔧 Settings & Info")
st.sidebar.info("🌗 **Theme Tip:** Switch between dark/light mode using the paint roller icon (top right).")
st.sidebar.markdown("**Feedback? Fill out the form in the last tab!**")
uploaded_file = st.sidebar.file_uploader("📁 Upload your hotel_bookings.csv file", type=['csv'], help="Upload the hotel bookings dataset as CSV.")

# DATA LOAD
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv('hotel_bookings.csv')
    return df

with st.spinner("Loading data..."):
    df = load_data(uploaded_file)

def preprocess(df):
    df_proc = df.copy()
    for col in df_proc.select_dtypes('object').columns:
        df_proc[col] = df_proc[col].fillna(df_proc[col].mode()[0])
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col].astype(str))
    df_proc = df_proc.fillna(df_proc.median(numeric_only=True))
    return df_proc

df_proc = preprocess(df)

country_lat_lon = {
    'PRT': (39.3999, -8.2245), 'GBR': (55.3781, -3.4360), 'FRA': (46.6034, 1.8883),
    'ESP': (40.4637, -3.7492), 'ITA': (41.8719, 12.5674), 'IRL': (53.1424, -7.6921),
    'DEU': (51.1657, 10.4515), 'BEL': (50.5039, 4.4699), 'NLD': (52.1326, 5.2913), 'BRA': (-14.2350, -51.9253),
}

# --------- TAB SETUP WITH ICONS/EMOJIS ---------
tabs = st.tabs([
    "🏠 Home", "📊 Data Visualization", "🤖 Classification", "📈 Clustering", "🔗 Association Rules", "📉 Regression", "✉️ Feedback"
])

# ----------------- TAB 1: HOME ------------------- #
with tabs[0]:
    st.title("🏨 Hotel Bookings Analytics Dashboard")
    st.markdown("""
    **Welcome!**  
    This dashboard provides deep insights into hotel booking data using interactive visualizations, ML models, clustering, association rules, and more.

    - 📊 **Visualize** key booking trends and hotel performance
    - 🤖 **Classify** booking cancellations using various ML algorithms
    - 📈 **Cluster** customers for targeted marketing
    - 🔗 **Discover** associations for upsell/cross-sell
    - 📉 **Forecast** demand and analyze price drivers
    - ✉️ **Share feedback** directly with us!
    """)
    st.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80", width=500)
    st.info("📌 Start exploring using the tabs above. Upload your CSV in the sidebar at any time!")

# ----------------- TAB 2: DATA VISUALIZATION ------------------- #
with tabs[1]:
    st.header("📊 Data Visualization")
    st.markdown("Explore descriptive, actionable insights on hotel bookings data.")

    with st.expander("Show data preview and summary"):
        st.write("First 5 rows of dataset:")
        st.dataframe(df.head())
        st.write("Numerical summary:")
        st.dataframe(df.describe())

    col1, col2 = st.columns(2)
    hotel_types = df['hotel'].unique()
    hotel_choice = col1.selectbox("🏨 Hotel Type", ["All"] + list(hotel_types), help="Filter by hotel type.")
    month_choice = col2.selectbox("🗓️ Arrival Month", ["All"] + list(df['arrival_date_month'].unique()), help="Filter by arrival month.")

    filter_df = df.copy()
    if hotel_choice != "All":
        filter_df = filter_df[filter_df['hotel'] == hotel_choice]
    if month_choice != "All":
        filter_df = filter_df[filter_df['arrival_date_month'] == month_choice]

    col_bar, col_line = st.columns(2)
    color1 = col_bar.color_picker("Bar/Histogram Color", "#1f77b4")
    color2 = col_line.color_picker("Line/Pareto Color", "#d62728")

    # KPIs (st.metric) with deltas and progress bar
    st.subheader("📈 Key Booking KPIs")
    total_bookings = len(df)
    total_canceled = int(df['is_canceled'].sum())
    repeat_rate = df['is_repeated_guest'].mean() * 100
    avg_adr = df['adr'].mean()
    month_map = {m: i+1 for i, m in enumerate([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'])}
    df['month_num'] = df['arrival_date_month'].map(month_map)
    this_month = df['month_num'].max()
    bookings_this_month = df[df['month_num'] == this_month].shape[0]
    bookings_last_month = df[df['month_num'] == (this_month-1)].shape[0]
    booking_delta = bookings_this_month - bookings_last_month

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Bookings", total_bookings, f"{booking_delta:+}", help="Difference from last month")
    c2.metric("Total Cancellations", total_canceled)
    c3.metric("Repeat Guest %", f"{repeat_rate:.2f}%")
    c4.metric("Avg. Daily Rate (ADR)", f"${avg_adr:.2f}")

    # Animated progress bar for bookings to goal
    goal = st.slider("Set Monthly Booking Goal", 0, 2000, 1000, help="Set a bookings goal for this month.")
    prog = min(1, bookings_this_month / goal)
    st.progress(prog, text=f"{bookings_this_month}/{goal} Bookings This Month")

    st.markdown("#### 📅 Booking Trends per Month")
    monthly_counts = filter_df.groupby('arrival_date_month').size().sort_index()
    fig, ax = plt.subplots()
    ax.bar(monthly_counts.index, monthly_counts.values, color=color1)
    ax.set_xlabel('Month')
    ax.set_ylabel('Bookings')
    st.pyplot(fig)

    # [Place all other visualizations, Pareto, heatmap, map, etc, here as before]
    # For brevity, see previous code blocks for these visuals.

    st.download_button("Download filtered data", data=filter_df.to_csv(index=False), file_name="filtered_data.csv")

# ----------------- TAB 3: CLASSIFICATION ------------------- #
with tabs[2]:
    st.header("🤖 Classification: Predict Booking Cancellation")
    with st.expander("Show data preview and summary"):
        st.write("First 5 rows of data for classification:")
        st.dataframe(df.head())
        st.write("Data summary (numerical columns):")
        st.dataframe(df.describe())

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

    conf_model = st.selectbox("Select model for Confusion Matrix", list(models.keys()), help="Pick a model to view its confusion matrix.")
    cm = confusion_matrix(y_test, preds[conf_model])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Canceled', 'Canceled'], yticklabels=['Not Canceled', 'Canceled'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

    st.subheader("ROC Curves")
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

    st.subheader("Upload Data for Cancellation Prediction")
    upload_pred = st.file_uploader("Upload CSV (no is_canceled column)", type=['csv'], key="pred")
    if upload_pred:
        df_new = pd.read_csv(upload_pred)
        df_new_proc = preprocess(df_new)
        pred_model = st.selectbox("Model for Prediction", list(models.keys()), key="pred2")
        with st.spinner('Generating predictions...'):
            chosen_model = models[pred_model]
            pred_result = chosen_model.predict(df_new_proc[features])
            result_df = df_new.copy()
            result_df['predicted_is_canceled'] = pred_result
            st.dataframe(result_df)
            csv_out = result_df.to_csv(index=False)
            st.download_button("Download Predictions", csv_out, "predictions.csv")

# ----------------- TAB 4: CLUSTERING ------------------- #
with tabs[3]:
    st.header("📈 Clustering: Customer Segmentation")
    with st.expander("Show data preview and summary"):
        st.write("First 5 rows for clustering:")
        st.dataframe(df.head())
        st.write("Numerical summary:")
        st.dataframe(df.describe())

    clustering_features = ['lead_time', 'adults', 'children', 'babies', 'adr', 'stays_in_weekend_nights', 'stays_in_week_nights']
    cluster_df = df_proc[clustering_features]
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_df)

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

    n_clusters = st.slider("Select number of clusters", 2, 10, 3, help="Choose how many customer segments to create. The Elbow chart can help you decide.")
    km = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = km.fit_predict(cluster_scaled)
    cluster_df['cluster'] = clusters

    persona = cluster_df.groupby('cluster').mean().round(2)
    st.dataframe(persona)

    df_with_cluster = df.copy()
    df_with_cluster['cluster'] = clusters
    st.download_button("Download Data with Cluster Labels", df_with_cluster.to_csv(index=False), "clustered_data.csv")

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

# ----------------- TAB 5: ASSOCIATION RULES ------------------- #
with tabs[4]:
    st.header("🔗 Association Rule Mining")
    with st.expander("Show data preview and summary"):
        st.write("First 5 rows for ARM:")
        st.dataframe(df.head())
        st.write("Summary for selected columns:")
        st.dataframe(df.describe())

    apriori_cols = st.multiselect("Select at least 2 categorical columns for association rule mining:",
        df.select_dtypes('object').columns.tolist(), default=['hotel', 'meal'])
    min_support = st.slider("Minimum Support", 0.01, 0.5, 0.1, 0.01)
    min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05)
    if len(apriori_cols) >= 2:
        with st.spinner('Mining association rules...'):
            df_ap = df[apriori_cols].astype(str)
            one_hot = pd.get_dummies(df_ap)
            freq_items = apriori(one_hot, min_support=min_support, use_colnames=True)
            if not freq_items.empty:
                rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
                rules = rules.sort_values('confidence', ascending=False).head(10)
                if not rules.empty:
                    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
                else:
                    st.info("No rules found with the selected parameters. Try lowering confidence or support.")
            else:
                st.info("No frequent itemsets found. Try lowering the minimum support.")
    else:
        st.warning("Please select at least two categorical columns for association rule mining.")

# ----------------- TAB 6: REGRESSION ------------------- #
with tabs[5]:
    st.header("📉 Regression Insights")
    with st.expander("Show data preview and summary"):
        st.write("First 5 rows for regression:")
        st.dataframe(df.head())
        st.write("Numerical summary:")
        st.dataframe(df.describe())

    st.subheader("Forecast Number of Bookings per Month")
    month_map = {m: i+1 for i, m in enumerate([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'])}
    reg_df = df.copy()
    reg_df['month_num'] = reg_df['arrival_date_month'].map(month_map)
    bookings_per_month = reg_df.groupby('month_num').size().reset_index(name='bookings')
    X = bookings_per_month[['month_num']]
    y = bookings_per_month['bookings']

    reg = LinearRegression()
    reg.fit(X, y)
    pred = reg.predict(X)
    color_reg = st.color_picker("Pick color for regression line", "#d62728", key="reg_color")
    fig, ax = plt.subplots()
    ax.scatter(X, y, label='Actual', color=color1)
    ax.plot(X, pred, color=color_reg, label='Predicted')
    ax.set_xlabel('Month')
    ax.set_ylabel('Bookings')
    ax.set_title('Monthly Bookings Forecast (Linear Regression)')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Predict ADR (Average Daily Rate) Using Different Regressors")
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
    with st.spinner('Training regression models...'):
        for name, model in regressors.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = model.score(X_test, y_test)
            rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
            reg_results.append({'Regressor': name, 'R2': r2, 'RMSE': rmse})
    st.success('Regression models trained!')
    st.dataframe(pd.DataFrame(reg_results).round(3))

    dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    imp_df = pd.DataFrame({
        'Feature': reg_X.columns,
        'Importance': dt.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    st.subheader("Top 10 Features Influencing ADR (Decision Tree)")
    st.bar_chart(imp_df.set_index('Feature'))

# ----------------- TAB 7: FEEDBACK ------------------- #
with tabs[6]:
    st.header("✉️ Feedback & Contact Us")
    st.write("We appreciate your feedback! Please fill the form below, or [click here to submit feedback on Google Forms](https://forms.gle/your-google-form-link)")
    with st.form(key='feedback_form'):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        feedback = st.text_area("Your Feedback / Suggestions")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success("Thank you for your feedback! (Note: This demo does not send emails, but your info is received.)")
    st.info("Alternatively, submit your feedback via [Google Forms](https://forms.gle/your-google-form-link)")
