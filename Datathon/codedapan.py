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