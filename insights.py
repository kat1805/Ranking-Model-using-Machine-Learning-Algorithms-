# This script will extract insights from the final rankings

import pandas as pd
import sys, json
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions as F
import seaborn as sns
#==============================================================================
# Create a spark session
spark = (
    SparkSession.builder.appName("MAST30034 Project 2 part 6")
    .config("spark.sql.repl.eagerEval.enabled", True) 
    .config("spark.sql.parquet.cacheMetadata", "true")
    .config("spark.sql.session.timeZone", "Etc/UTC")
    .config("spark.driver.memory", "10g")
    .getOrCreate()
)
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
# top 100 merchants overall
top_100 = pd.read_csv(curated_data_path + "final_rank.csv").iloc[:,1:]

final_100_sdf = spark.createDataFrame(top_100)
final_100_sdf.createOrReplaceTempView('top100')

#------------------------------------------------------------------------------
# categories of top 100 merchants

agg_top = spark.sql(""" 

SELECT category, COUNT(merchant_abn) AS no_of_merchants
FROM top100
GROUP BY category
ORDER BY no_of_merchants DESC
""")

top_categories = agg_top.toPandas()

#------------------------------------------------------------------------------
# Donut chart: categories of top 100 merchants

data = top_categories['no_of_merchants']
category_labels = top_categories['category']
 
# Create a pie plot
plt.pie(data, labels=category_labels, autopct='%1.1f%%', pctdistance=0.85)

# add a circle at the center to transform it in a donut chart
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.savefig(visualisation_path + "pie chart.jpg", bbox_inches='tight')
plt.show()


total_revenue = top_100[['total_earnings_of_BNPL', 'total_future_transactionss'
,'total_future_customers']]
total_revenue.columns = ["BNPL earnings", "Transactions", "Customers"]
fig1, ax1 = plt.subplots(figsize=(12,7))
sns.boxplot(data=total_revenue, orient="h",dodge=False)
ax1.set_xlabel("Count")
ax1.set_title("Distrbution of different features for the top 100 merchants")
plt.savefig(visualisation_path + "Gender transactions.jpg",dpi=300, 
bbox_inches='tight')
plt.savefig(visualisation_path + "top 100 distribution.jpg")
