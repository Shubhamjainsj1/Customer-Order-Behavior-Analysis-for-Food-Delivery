# Customer-Order-Behavior-Analysis-for-Food-Delivery

# 🍽️ Swiggy Customer Order Behavior Dashboard

An interactive data visualization app built with **Streamlit** that analyzes customer behavior and restaurant data from the Swiggy platform.

## 📊 Features

- **Filter** by city and food type
- **View insights** like:
  - Top restaurants by total ratings
  - Average price and delivery time
  - Popular food types
  - Price vs. Rating scatter plot
  - Delivery time distribution
- **City-wise summary** of average ratings, pricing, and restaurant count

## 📁 Dataset

The dataset contains details such as:

| Column Name       | Description                         |
|------------------|-------------------------------------|
| `ID`             | Unique restaurant/order ID          |
| `Area`           | Area of operation                   |
| `City`           | City where the restaurant is located|
| `Restaurant`     | Restaurant name                     |
| `Price`          | Average price for an order          |
| `Avg ratings`    | Average customer rating             |
| `Total ratings`  | Total number of ratings             |
| `Food type`      | Cuisine or food category            |
| `Address`        | Full restaurant address             |
| `Delivery time`  | Estimated delivery time (in minutes)|

📌 Source: [Kaggle - Swiggy Restaurant Dataset](https://www.kaggle.com/datasets/abhijitdahatonde/swiggy-restuarant-dataset)

---

## 🚀 Installation

1. **Clone the repository**

```bash
git clone https://github.com/Shubhamjainsj1/Customer-Order-Behavior-Analysis-for-Food-Delivery.git
cd Customer-Order-Behavior-Analysis-for-Food-Delivery
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
````
 ***If requirements.txt is missing, install manually:***
```bash
pip install streamlit pandas matplotlib seaborn
````
3.  **Run the app**
```bash
streamlit run main.py
```
4. **Open browser at http://localhost:8501**
  ---
## Link : https://customer-order-behavior-analysis-for-food-delivery.streamlit.app/
