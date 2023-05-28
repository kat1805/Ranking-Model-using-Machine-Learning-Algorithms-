#This script visualise the final ranking output for the top 100 merchants.

import pandas as pd
import sys, json
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Define relative target directories

paths_arg = sys.argv[1]

with open(paths_arg) as json_paths: 
    PATHS = json.load(json_paths)
    json_paths.close()

raw_internal_path = PATHS['raw_internal_data_path']
curated_data_path = PATHS['curated_data_path']
external_data_path = PATHS['external_data_path']
visualisation_path = PATHS['visualisation_path']
#------------------------------------------------------------------------------

# top 100 overall
rank = pd.read_csv(curated_data_path + "final_rank.csv").iloc[:,1:]
rank['rank'] = rank.index
rank["total_revenue (in hundred)"] = rank["total_revenue"] / 100


# rank for each categories
# category for the merchants
category_labels = rank.groupby(by="category").count().index

# rank for each category contained in the top 100 merchants result
rank_category = [rank[rank['category'] == cat] for cat in category_labels]

# order to display on the graph: from top to bottom, it shows rank 1-5 merchants
rank_order = [4,3,2,1,0]

# Top 5 merchants 
# calculate the average
avg_rank = rank.loc[:,["total_revenue (in hundred)", "total_future_customers", 
"total_earnings_of_BNPL", "total_future_transactionss", 
"average_fraud_rate_per_merchant"]].mean()
avg_rank["merchant_name"] = "average of top 100"

# get top 5 merchants
data = rank.loc[rank_order,["merchant_name", "total_revenue (in hundred)", 
"total_future_customers", "total_earnings_of_BNPL", 
"total_future_transactionss", "average_fraud_rate_per_merchant"]]

# add average of top 100 merchants for comparison
data = pd.concat([data, avg_rank.to_frame().T], ignore_index = True)

# label by merchant name
data = data.rename(columns={"merchant_name": "merchant"}).set_index("merchant")
data.plot.barh(title=f"Top 5 merchants").legend(bbox_to_anchor=(1.01, 1), 
loc='upper left', borderaxespad=0)

plt.savefig(visualisation_path + "top5.jpg", bbox_inches="tight")



# Top 5 merchants in "Beauty, Health, Personal and Household"
# make sure it is ordered by the rank 
data0 = rank_category[0].sort_values("rank").reset_index(drop=True)

# category name
category = data0["category"][0]

# calculate the average
avg_rank0 = data0.loc[:,["total_revenue (in hundred)", "total_future_customers", 
"total_earnings_of_BNPL", "total_future_transactionss", 
"average_fraud_rate_per_merchant"]].mean()
avg_rank0["merchant_name"] = f"average of merchants in\n {category}"

# top 5 wihtin the category
data0 = data0.loc[rank_order,["merchant_name", "total_revenue (in hundred)", 
"total_future_customers", "total_earnings_of_BNPL", 
"total_future_transactionss", "average_fraud_rate_per_merchant"]]

# add average of merchants with category "Beauty, Health, Personal and 
# Household" with rank above 100
data0 = pd.concat([data0, avg_rank0.to_frame().T], ignore_index = True)

# label by merchant name
data0 = data0.rename(columns={"merchant_name": "merchant"}).set_index("merchant")

data0.plot.barh(title=f"Top 5 merchants in {category}").legend(
    bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

plt.savefig(visualisation_path + "top5_beauty.jpg", bbox_inches="tight")



# Top 5 merchants in "Books, Stationary and Music"
# make sure it is ordered by the rank 
data1 = rank_category[1].sort_values("rank").reset_index(drop=True)

# category name
category = data1["category"][0]

# calculate the average
avg_rank1 = data1.loc[:,["total_revenue (in hundred)", "total_future_customers", 
"total_earnings_of_BNPL", "total_future_transactionss", 
"average_fraud_rate_per_merchant"]].mean()
avg_rank1["merchant_name"] = f"average of merchants in\n {category}"

data1 = data1.loc[rank_order,["merchant_name", "total_revenue (in hundred)", 
"total_future_customers", "total_earnings_of_BNPL","total_future_transactionss", 
"average_fraud_rate_per_merchant"]]

# add average of merchants with category "Books, Stationary and Music" with rank above 100
data1 = pd.concat([data1, avg_rank1.to_frame().T], ignore_index = True)

# label by merchant name
data1 = data1.rename(columns={"merchant_name": "merchant"}).set_index("merchant")

data1.plot.barh(title=f"Top 5 merchants in {category}").legend(
    bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

plt.savefig(visualisation_path + "top5_books.jpg", bbox_inches="tight")



# Top 5 merchants in "Electronics"
# make sure it is ordered by the rank 
data2 = rank_category[2].sort_values("rank").reset_index(drop=True)

# category name
category = data2["category"][0]

# calculate the average
avg_rank2 = data2.loc[:,["total_revenue (in hundred)", "total_future_customers", 
"total_earnings_of_BNPL", "total_future_transactionss", 
"average_fraud_rate_per_merchant"]].mean()
avg_rank2["merchant_name"] = f"average of merchants in\n {category}"

data2 = data2.loc[rank_order,["merchant_name", "total_revenue (in hundred)", 
"total_future_customers", "total_earnings_of_BNPL", "total_future_transactionss", 
"average_fraud_rate_per_merchant"]]

# add average of merchants with category "Electronics" with rank above 100
data2 = pd.concat([data2, avg_rank2.to_frame().T], ignore_index = True)

# label by merchant name
data2 = data2.rename(columns={"merchant_name": "merchant"}).set_index("merchant")

data2.plot.barh(title=f"Top 5 merchants in {category}").legend(
    bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

plt.savefig(visualisation_path + "top5_electronics.jpg", bbox_inches="tight")



# Top 5 merchants in "Furniture"
# make sure it is ordered by the rank 
data3 = rank_category[3].sort_values("rank").reset_index(drop=True)

# category name
category = data3["category"][0]

# calculate the average
avg_rank3 = data3.loc[:,["total_revenue (in hundred)", "total_future_customers", 
"total_earnings_of_BNPL", "total_future_transactionss", 
"average_fraud_rate_per_merchant"]].mean()
avg_rank3["merchant_name"] = f"average of merchants in\n {category}"

data3 = data3.loc[rank_order,["merchant_name", "total_revenue (in hundred)", 
"total_future_customers", "total_earnings_of_BNPL", "total_future_transactionss", 
"average_fraud_rate_per_merchant"]]

# add average of merchants with category "Furniture" with rank above 100
data3 = pd.concat([data3, avg_rank3.to_frame().T], ignore_index = True)

# label by merchant name
data3 = data3.rename(columns={"merchant_name": "merchant"}).set_index("merchant")

data3.plot.barh(title=f"Top 5 merchants in {category}").legend(
    bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

plt.savefig(visualisation_path + "top5_furniture.jpg", bbox_inches="tight")



# Top 5 merchants in "Toys and DIY"
# make sure it is ordered by the rank 
data4 = rank_category[4].sort_values("rank").reset_index(drop=True)

# category name
category = data4["category"][0]

# calculate the average
avg_rank4 = data4.loc[:,["total_revenue (in hundred)", "total_future_customers", 
"total_earnings_of_BNPL", "total_future_transactionss", 
"average_fraud_rate_per_merchant"]].mean()
avg_rank4["merchant_name"] = f"average of merchants in\n {category}"

data4 = data4.loc[rank_order,["merchant_name", "total_revenue (in hundred)", 
"total_future_customers", "total_earnings_of_BNPL", "total_future_transactionss", 
"average_fraud_rate_per_merchant"]]

# add average of merchants with category "Toys and DIY" with rank above 100
data4 = pd.concat([data4, avg_rank4.to_frame().T], ignore_index = True)

# label by merchant name
data4 = data4.rename(columns={"merchant_name": "merchant"}).set_index("merchant")

data4.plot.barh(title=f"Top 5 merchants in {category}").legend(
    bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

plt.savefig(visualisation_path + "top5_toys.jpg", bbox_inches="tight")