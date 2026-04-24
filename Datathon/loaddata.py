import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "."


def load_data():
    data = {}

    data["products"] = pd.read_csv(os.path.join(DATA_PATH, "products.csv"))
    data["customers"] = pd.read_csv(os.path.join(DATA_PATH, "customers.csv"))
    data["promotions"] = pd.read_csv(os.path.join(DATA_PATH, "promotions.csv"))
    data["geography"] = pd.read_csv(os.path.join(DATA_PATH, "geography.csv"))

    data["orders"] = pd.read_csv(
        os.path.join(DATA_PATH, "orders.csv"),
        parse_dates=["order_date"]
    )

    data["order_items"] = pd.read_csv(os.path.join(DATA_PATH, "order_items.csv"))
    data["payments"] = pd.read_csv(os.path.join(DATA_PATH, "payments.csv"))

    data["shipments"] = pd.read_csv(
        os.path.join(DATA_PATH, "shipments.csv"),
        parse_dates=["ship_date", "delivery_date"]
    )

    data["returns"] = pd.read_csv(
        os.path.join(DATA_PATH, "returns.csv"),
        parse_dates=["return_date"]
    )

    data["reviews"] = pd.read_csv(
        os.path.join(DATA_PATH, "reviews.csv"),
        parse_dates=["review_date"]
    )

    data["sales"] = pd.read_csv(
        os.path.join(DATA_PATH, "sales.csv"),
        parse_dates=["Date"]
    )

    data["inventory"] = pd.read_csv(
        os.path.join(DATA_PATH, "inventory.csv"),
        parse_dates=["snapshot_date"]
    )

    data["web_traffic"] = pd.read_csv(
        os.path.join(DATA_PATH, "web_traffic.csv"),
        parse_dates=["date"]
    )

    data["sample_submission"] = pd.read_csv(
        os.path.join(DATA_PATH, "sample_submission.csv")
    )

    return data


def build_gold_dataset(data):
    gold_df = (
        data["order_items"]
        .merge(data["orders"], on="order_id", how="left")
        .merge(data["products"], on="product_id", how="left")
        .merge(data["customers"], on="customer_id", how="left", suffixes=("", "_customer"))
        .merge(data["promotions"], on="promo_id", how="left", suffixes=("", "_promo"))
    )

    gold_df["gross_revenue"] = gold_df["quantity"] * gold_df["unit_price"]
    gold_df["net_revenue"] = gold_df["gross_revenue"] - gold_df["discount_amount"]
    gold_df["cogs_total"] = gold_df["quantity"] * gold_df["cogs"]
    gold_df["profit"] = gold_df["net_revenue"] - gold_df["cogs_total"]

    gold_df["profit_margin"] = gold_df["profit"] / gold_df["net_revenue"]

    # SỬA QUAN TRỌNG:
    # Có khuyến mãi nếu promo_id hoặc promo_id_2 tồn tại
    gold_df["has_promo"] = (
        gold_df["promo_id"].notnull() |
        gold_df["promo_id_2"].notnull()
    )

    return gold_df


def basic_analysis(data, gold_df):
    print("\n========== GOLD DATASET CHECK ==========")
    print("Shape:", gold_df.shape)
    print("Duplicate rows:", gold_df.duplicated().sum())

    print("\nPromo line count:")
    print(gold_df["has_promo"].value_counts())

    print("\nPromo line percentage:")
    print(gold_df["has_promo"].value_counts(normalize=True) * 100)

    print("\nPromo vs Non-Promo Average Revenue:")
    print(gold_df.groupby("has_promo")["net_revenue"].mean())

    print("\nPromo vs Non-Promo Total Revenue:")
    print(gold_df.groupby("has_promo")["net_revenue"].sum())

    print("\nPromo vs Non-Promo Average Profit:")
    print(gold_df.groupby("has_promo")["profit"].mean())

    print("\nPromo vs Non-Promo Average Profit Margin:")
    print(gold_df.groupby("has_promo")["profit_margin"].mean())


def plot_daily_revenue(gold_df):
    daily_revenue = (
        gold_df.groupby("order_date")["net_revenue"]
        .sum()
        .sort_index()
    )

    plt.figure(figsize=(12, 6))
    daily_revenue.plot()
    plt.title("Daily Revenue Over Time")
    plt.xlabel("Date")
    plt.ylabel("Net Revenue")
    plt.tight_layout()
    plt.show()


def plot_promo_vs_nonpromo(gold_df):
    promo_daily = (
        gold_df.groupby(["order_date", "has_promo"])["net_revenue"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

    plt.figure(figsize=(12, 6))

    if False in promo_daily.columns:
        plt.plot(promo_daily.index, promo_daily[False], label="Non-Promo Revenue")

    if True in promo_daily.columns:
        plt.plot(promo_daily.index, promo_daily[True], label="Promo Revenue")

    plt.title("Daily Revenue: Promo vs Non-Promo")
    plt.xlabel("Date")
    plt.ylabel("Net Revenue")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_promo_revenue_share(gold_df):
    promo_share = (
        gold_df.groupby("has_promo")["net_revenue"]
        .sum()
        .rename(index={False: "Non-Promo", True: "Promo"})
    )

    plt.figure(figsize=(7, 5))
    promo_share.plot(kind="bar")
    plt.title("Total Revenue Share: Promo vs Non-Promo")
    plt.xlabel("Order Type")
    plt.ylabel("Total Net Revenue")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_promo_profit_margin(gold_df):
    promo_margin = (
        gold_df.groupby("has_promo")["profit_margin"]
        .mean()
        .rename(index={False: "Non-Promo", True: "Promo"})
    )

    plt.figure(figsize=(7, 5))
    promo_margin.plot(kind="bar")
    plt.title("Average Profit Margin: Promo vs Non-Promo")
    plt.xlabel("Order Type")
    plt.ylabel("Average Profit Margin")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = load_data()
    gold_df = build_gold_dataset(data)

    basic_analysis(data, gold_df)

    plot_daily_revenue(gold_df)
    plot_promo_vs_nonpromo(gold_df)
    plot_promo_revenue_share(gold_df)
    plot_promo_profit_margin(gold_df)

    print("\nDONE: Promotion logic updated using promo_id and promo_id_2.")
    # ==============================
# Q1: Inter-order gap
# ==============================

orders = data["orders"]

# sort theo customer + date
orders_sorted = orders.sort_values(["customer_id", "order_date"])

# tính khoảng cách giữa các order
orders_sorted["prev_date"] = orders_sorted.groupby("customer_id")["order_date"].shift(1)
orders_sorted["gap_days"] = (orders_sorted["order_date"] - orders_sorted["prev_date"]).dt.days

# chỉ lấy khách có nhiều hơn 1 order
valid_gaps = orders_sorted["gap_days"].dropna()

median_gap = valid_gaps.median()

print("\nQ1 - Median inter-order gap:", median_gap)

# ==============================
# Q2: Profit margin by segment
# ==============================

products = data["products"].copy()

# tính margin
products["margin"] = (products["price"] - products["cogs"]) / products["price"]

# group theo segment
segment_margin = products.groupby("segment")["margin"].mean().sort_values(ascending=False)

print("\nQ2 - Average margin by segment:")
print(segment_margin)
# ==============================
# Q3: Return reason (Streetwear)
# ==============================

returns = data["returns"]
products = data["products"]

# join returns với products
returns_merged = returns.merge(products, on="product_id", how="left")

# lọc category Streetwear
streetwear_returns = returns_merged[returns_merged["category"] == "Streetwear"]

# đếm lý do trả hàng
reason_counts = streetwear_returns["return_reason"].value_counts()

print("\nQ3 - Return reasons for Streetwear:")
print(reason_counts)

# ==============================
# Q4: Lowest bounce rate
# ==============================

traffic = data["web_traffic"]

bounce_avg = traffic.groupby("traffic_source")["bounce_rate"].mean().sort_values()

print("\nQ4 - Average bounce rate by traffic source:")
print(bounce_avg)
# ==============================
# Q5: Promo usage rate
# ==============================

order_items = data["order_items"]

promo_rate = order_items["promo_id"].notnull().mean() * 100

print("\nQ5 - Promo usage rate (%):", promo_rate)
# ==============================
# Q6: Avg orders per customer by age_group
# ==============================

customers = data["customers"]
orders = data["orders"]

# lọc age_group không null
customers_valid = customers[customers["age_group"].notnull()]

# join với orders
merged = customers_valid.merge(orders, on="customer_id", how="left")

# số đơn mỗi nhóm
orders_per_group = merged.groupby("age_group")["order_id"].count()

# số khách mỗi nhóm
customers_per_group = customers_valid.groupby("age_group")["customer_id"].nunique()

# tính trung bình
avg_orders = (orders_per_group / customers_per_group).sort_values(ascending=False)

print("\nQ6 - Avg orders per customer by age_group:")
print(avg_orders)
# ==============================
# Q7: Revenue by region (CORRECT)
# ==============================

orders = data["orders"]
geography = data["geography"]
order_items = data["order_items"]

# join order_items với orders
df = order_items.merge(orders, on="order_id", how="left")

# join với geography để lấy region
df = df.merge(geography, on="zip", how="left")

# tính revenue thật
df["revenue"] = df["quantity"] * df["unit_price"] - df["discount_amount"]

# group theo region
region_revenue = df.groupby("region")["revenue"].sum().sort_values(ascending=False)

print("\nQ7 - Total revenue by region (CORRECT):")
print(region_revenue)

# ==============================
# Q8: Payment method for cancelled orders
# ==============================

orders = data["orders"]
payments = data["payments"]

cancelled = orders[orders["order_status"] == "cancelled"]

df_cancel = cancelled.merge(
    payments,
    on="order_id",
    how="left",
    suffixes=("_order", "_payment")
)

print("\nColumns after merge:")
print(df_cancel.columns)

result = df_cancel["payment_method_payment"].value_counts()

print("\nQ8 - Payment methods for cancelled orders:")
print(result)

# ==============================
# Q9 FIX: dùng return_quantity
# ==============================

order_items = data["order_items"]
products = data["products"]
returns = data["returns"]

# order_items + size
oi = order_items.merge(products[["product_id", "size"]], on="product_id", how="left")

# returns + size
ret = returns.merge(products[["product_id", "size"]], on="product_id", how="left")

# tổng số item bán
total_by_size = oi.groupby("size")["quantity"].sum()

# tổng số item bị trả
returns_by_size = ret.groupby("size")["return_quantity"].sum()

# return rate đúng
return_rate = (returns_by_size / total_by_size)

# chỉ lấy S M L XL
valid_sizes = ["S", "M", "L", "XL"]
return_rate = return_rate.loc[return_rate.index.isin(valid_sizes)]

# sort
return_rate = return_rate.sort_values(ascending=False)

print("\nQ9 - Return rate by size (CORRECT):")
print(return_rate)

# ==============================
# Q10: Average payment value by installments
# ==============================

payments = data["payments"]

avg_payment_by_installment = (
    payments.groupby("installments")["payment_value"]
    .mean()
    .sort_values(ascending=False)
)

print("\nQ10 - Average payment value by installments:")
print(avg_payment_by_installment)