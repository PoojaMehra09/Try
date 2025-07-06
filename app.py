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
import time

st.set_page_config(page_title="🏨 Hotel Bookings Dashboard", layout="wide")

# Hide Streamlit menu/footer for a cleaner look
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- SIDEBAR
st.sidebar.title("🔧 Settings & Info")
st.sidebar.info(
    "🌗 **Theme Tip:** You can switch between dark/light mode using the paint roller icon (top right of the page)."
)
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

    # --- FILTERS
    col1, col2 = st.columns(2)
    hotel_types = df['hotel'].unique()
    hotel_choice = col1.selectbox("🏨 Hotel Type", ["All"] + list(hotel_types), help="Filter by hotel type.")
    month_choice = col2.selectbox("🗓️ Arrival Month", ["All"] + list(df['arrival_date_month'].unique()), help="Filter by arrival month.")

    filter_df = df.copy()
    if hotel_choice != "All":
        filter_df = filter_df[filter_df['hotel'] == hotel_choice]
    if month_choice != "All":
        filter_df = filter_df[filter_df['arrival_date_month'] == month_choice]

    # --- COLOR PICKERS
    col_bar, col_line = st.columns(2)
    color1 = col_bar.color_picker("Bar/Histogram Color", "#1f77b4")
    color2 = col_line.color_picker("Line/Pareto Color", "#d62728")

    # --- KPIs (st.metric) with deltas and progress bar ---
    st.subheader("📈 Key Booking KPIs")
    total_bookings = len(df)
    total_canceled = int(df['is_canceled'].sum())
    repeat_rate = df['is_repeated_guest'].mean() * 100
    avg_adr = df['adr'].mean()
    # Calculate month-over-month booking trend (delta)
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

    # --- Animated progress bar for bookings to goal ---
    goal = st.slider("Set Monthly Booking Goal", 0, 2000, 1000, help="Set a bookings goal for this month.")
    prog = min(1, bookings_this_month / goal)
    st.progress(prog, text=f"{bookings_this_month}/{goal} Bookings This Month")

    # --- REST OF VISUALIZATIONS (unchanged, use previous code) ---
    st.markdown("#### 📅 Booking Trends per Month")
    monthly_counts = filter_df.groupby('arrival_date_month').size().sort_index()
    fig, ax = plt.subplots()
    ax.bar(monthly_counts.index, monthly_counts.values, color=color1)
    ax.set_xlabel('Month')
    ax.set_ylabel('Bookings')
    st.pyplot(fig)

    # [Your other visualizations here...]

    # Download filtered data
    st.download_button("Download filtered data", data=filter_df.to_csv(index=False), file_name="filtered_data.csv")

# ------------- THE REST OF YOUR TABS... -----------
# [For brevity, all previous tab code for Classification, Clustering, Assoc, Regression remains the same
# -- just use the same code as before, only with icons/emojis in tab and header names as above!]

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
            # Here you could also send this info to Google Sheets, email, or save to file/database.

    st.info("Alternatively, submit your feedback via [Google Forms](https://forms.gle/your-google-form-link)")

