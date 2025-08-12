
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

# -----------------------------
# 🧠 PHASE 1: Predict Delivery Time using XGBoost
# -----------------------------
st.subheader("📈 Predict Delivery Time (Regression Model)")

if st.button("Train Delivery Time Model"):
    from sklearn.model_selection import train_test_split,RandomizedSearchCV
    from sklearn.metrics import mean_squared_error
    import xgboost as xgb
    import joblib
    import math
    import os


    # Prepare data for modeling
    ml_df = filtered_df.copy()
    ml_df.dropna(subset=['Price', 'Avg ratings', 'Total ratings', 'Delivery time'], inplace=True)

    # Encode categorical variables
    ml_encoded = pd.get_dummies(ml_df[['City', 'Food type']], drop_first=True)

    # Feature matrix and target
    X = pd.concat([ml_df[['Price', 'Avg ratings', 'Total ratings']], ml_encoded], axis=1)
    y = ml_df['Delivery time']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0]
    }

    base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=10,
        scoring='neg_root_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Save the trained model and ensure the directory exists

    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, "models/delivery_time_xgb.pkl")

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    st.success(f"✅ Model trained with tuning. RMSE: {rmse:.2f} minutes")
    st.info(f"Best Parameters: {search.best_params_}")

    # Feature importance
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(10)

    st.write("🔍 **Top 10 Important Features**")
    fig, ax = plt.subplots()
    sns.barplot(x=top_features.values, y=top_features.index, ax=ax)
    st.pyplot(fig)
