import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("swiggy.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Handle missing or malformed values
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Avg ratings'] = pd.to_numeric(df['Avg ratings'], errors='coerce')
    df['Total ratings'] = pd.to_numeric(df['Total ratings'], errors='coerce')
    df['Delivery time'] = pd.to_numeric(df['Delivery time'], errors='coerce')

    df['Food type'] = df['Food type'].astype(str).str.strip()
    df['City'] = df['City'].astype(str).str.strip()
    df['Restaurant'] = df['Restaurant'].astype(str).str.strip()

    # Drop rows with missing essential data
    df.dropna(subset=['Price', 'Avg ratings', 'City', 'Restaurant'], inplace=True)
    return df

df = load_data()

# Title
st.title("🍽️ Swiggy Customer Order Behavior Dashboard")

# Sidebar filters
st.sidebar.header("📍 Filter Options")
selected_city = st.sidebar.multiselect("Select Cities", df['City'].unique(), default=df['City'].unique())
selected_food_type = st.sidebar.multiselect("Select Food Types", df['Food type'].unique(), default=df['Food type'].unique())

filtered_df = df[
    (df['City'].isin(selected_city)) &
    (df['Food type'].isin(selected_food_type))
]

# Summary Stats
st.subheader("📊 Summary Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Restaurants", filtered_df['Restaurant'].nunique())
col2.metric("Avg Rating", f"{filtered_df['Avg ratings'].mean():.2f}")
col3.metric("Avg Price", f"₹{filtered_df['Price'].mean():.0f}")

# Top Restaurants by Total Ratings
st.subheader("🏆 Top Rated Restaurants (by Total Ratings)")
top_rated = filtered_df.groupby('Restaurant')['Total ratings'].sum().sort_values(ascending=False).head(10)
st.bar_chart(top_rated)

# Price vs Rating Scatter
st.subheader("💲 Price vs Rating")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=filtered_df.sample(min(1000, len(filtered_df))), x='Price', y='Avg ratings', hue='City', alpha=0.6, ax=ax1)
ax1.set_xlabel("Price (₹)")
ax1.set_ylabel("Average Rating")
st.pyplot(fig1)

# Food Type Popularity
st.subheader("🍔 Popular Food Types")
food_counts = filtered_df['Food type'].value_counts().head(10)
st.bar_chart(food_counts)

# Delivery Time Distribution
st.subheader("⏱️ Delivery Time Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df['Delivery time'], bins=20, kde=True, ax=ax2)
ax2.set_xlabel("Delivery Time (mins)")
st.pyplot(fig2)

# City-wise stats
st.subheader("📍 City-wise Summary")
city_stats = filtered_df.groupby('City').agg({
    'Restaurant': 'nunique',
    'Avg ratings': 'mean',
    'Price': 'mean',
    'Delivery time': 'mean'
}).rename(columns={
    'Restaurant': 'Num Restaurants',
    'Avg ratings': 'Avg Rating',
    'Price': 'Avg Price',
    'Delivery time': 'Avg Delivery Time'
})
st.dataframe(city_stats.sort_values("Num Restaurants", ascending=False))

# Optionally show raw data
if st.checkbox("📄 Show raw data"):
    st.dataframe(filtered_df)
