import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
import lbl2vec
import sys, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import folium
import matplotlib
from pyspark.sql.functions import date_format
import seaborn as sns
#==============================================================================
# Create a spark session
#==============================================================================
spark = (
    SparkSession.builder.appName("MAST30034 Project 1")
    .config("spark.sql.repl.eagerEval.enabled", True) 
    .config("spark.sql.parquet.cacheMetadata", "true")
    .config("spark.sql.session.timeZon", "Etc/UTC")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "8g")
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
transactions = spark.read.parquet(curated_data_path + "full_join.parquet")

covid_transaction_data = transactions.select("merchant_abn", "order_id", 
"order_datetime")

covid_transaction_data = covid_transaction_data.select(F.col("order_datetime"
),\
F.to_date(F.col("order_datetime"),"MM-yyyy").alias("temp_date"),
F.col("order_id"))

covid_transaction_data = covid_transaction_data.withColumn\
("month", F.month(F.col("temp_date")))
covid_transaction_data = covid_transaction_data.withColumn\
("year", F.year(F.col("temp_date")))

transactions_2021 = covid_transaction_data.filter(
    covid_transaction_data.year == 2021)
transactions_2022 = covid_transaction_data.filter(
    covid_transaction_data.year == 2021)

transactions_month_2021 = transactions_2021.groupBy("month").agg(
    F.count("order_id").\
alias("num_transactions_month"))

transactions_month_2022 = transactions_2022.groupBy("month").agg(
    F.count("order_id").\
alias("num_transactions_month"))

covid_case_data = pd.read_csv(curated_data_path + "covid.csv")
covid_case_sdf = spark.createDataFrame(covid_case_data)

#Inner join:

covid_transaction_data.createOrReplaceTempView("temp")

covid_case_sdf.createOrReplaceTempView("temp2")

covid_plot_data = spark.sql("""

SELECT *
FROM temp


INNER JOIN temp2

ON temp.temp_date = temp2.date
""")

#Filtering
data_2021 = covid_plot_data.filter(covid_plot_data.year == 2021)
data_2022 = covid_plot_data.filter(covid_plot_data.year == 2022)

aggregated_data_2021 = data_2021.groupBy("mm").agg(F.sum("covid_cases").\
alias("total_covid_cases_month"))

aggregated_data_2022 = data_2022.groupBy("mm").agg(F.sum("covid_cases").\
alias("total_covid_cases_month"))

#Joins

#2021

transactions_month_2021.createOrReplaceTempView("temp")

aggregated_data_2021.createOrReplaceTempView("temp2")

covid_plot_data_2021 = spark.sql("""

SELECT *
FROM temp


INNER JOIN temp2

ON temp.month = temp2.mm
""")

#2022

transactions_month_2021.createOrReplaceTempView("temp")

aggregated_data_2022.createOrReplaceTempView("temp2")

covid_plot_data_2022 = spark.sql("""

SELECT *
FROM temp


INNER JOIN temp2

ON temp.month = temp2.mm
""")

covid_plot_data_2021_pdf = covid_plot_data_2021.toPandas()
covid_plot_data_2022_pdf = covid_plot_data_2022.toPandas()

# covid_plot_data_2021_pdf.plot(x='total_covid_cases_month', 
# y='num_transactions_month', style='o')
# covid_plot_data_2022_pdf.plot(x='total_covid_cases_month', 
# y='num_transactions_month', style='o')

# covid_plot_2021 = sns.scatterplot(data=covid_plot_data_2021_pdf, 
# x="total_covid_cases_month", y="num_transactions_month")
# plt.savefig("../plots/covid_transactions_2021.jpg")

# covid_plot_2022 = sns.scatterplot(data=covid_plot_data_2022_pdf, 
# x="total_covid_cases_month", y="num_transactions_month")
# plt.savefig("../plots/covid_transactions_2022.jpg")

plt.figure()
plt.scatter(covid_plot_data_2021_pdf['total_covid_cases_month'],
covid_plot_data_2021_pdf['num_transactions_month'])
plt.title('Covid Cases per Month vs Number of Transactions per Month')
plt.xlabel('Total Covid Cases per Month')
plt.ylabel('Number of Transactions per Month')
plt.savefig(visualisation_path + "covid_transactions_2021.jpg", 
dpi=300, bbox_inches='tight')

plt.figure().clear()

plt.figure()
plt.scatter(covid_plot_data_2022_pdf['total_covid_cases_month'],
covid_plot_data_2022_pdf['num_transactions_month'])
plt.title('Covid Cases per Month vs Number of Transactions per Month')
plt.xlabel('Total Covid Cases per Month')
plt.ylabel('Number of Transactions per Month')
plt.savefig(visualisation_path + "covid_transactions_2022.jpg",
dpi=300, bbox_inches='tight')




