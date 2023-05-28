#==============================================================================
import pandas as pd
import sys, json
from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.sql.functions import when
from pyspark.sql import SparkSession
from pyspark.sql import functions as countDistinct

#==============================================================================
# Create a spark session
#==============================================================================
spark = (
    SparkSession.builder.appName("MAST30034 Project 2")
    .config("spark.sql.repl.eagerEval.enabled", True)
    .config("spark.sql.parquet.cacheMetadata", "true")
    .config("spark.sql.session.timeZone", "Etc/UTC")
    .config("spark.driver.memory", "15g")
    .config("spark.executor.memory", "5g")
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


#==============================================================================
# LOAD THE DATA
#==============================================================================
# Retrive the required data

segments_data = spark.read.option("header","true").csv( 
    curated_data_path + "tagged_merchants.csv")
full_join = spark.read.parquet(curated_data_path + "full_join.parquet")

#==============================================================================
# JOINING THE DATA TABLES
#==============================================================================

#Joining the full dataset dataset with segments
segments_data = segments_data.withColumnRenamed("name", "merchant_name1")\
       .withColumnRenamed("merchant_abn", "merchant_abn1")


full_join.createOrReplaceTempView("temp")

segments_data.createOrReplaceTempView("temp2")

final_join = spark.sql("""
SELECT *
FROM temp
INNER JOIN temp2
ON temp.merchant_abn = temp2.merchant_abn1
""")

#==============================================================================
# EVALUATING AVERAGE FRAUD PROBABILITIES
#==============================================================================

#Creating a table of average fraud probabilities for each category of purchase:

aggregated_prob = final_join.groupBy("temp2.category").agg(
    F.avg("fraud_probability_consumer").\
alias("avg_consumer_fraud"),F.avg(
    "fraud_probability_merchant").alias("avg_merchant_fraud"))

#adding a column for is_fraud BOOLEAN value for each transaction:

is_fraud = when((final_join["category"] == "Electronics") & (
    final_join["fraud_probability_consumer"] > 15.085)&(
    final_join["fraud_probability_merchant"]>29.635),1)\
    .when((final_join["category"] == "Toys and DIY") & (
    final_join["fraud_probability_consumer"] > 15.843) &(
    final_join["fraud_probability_merchant"]>32.404),1)\
    .when((final_join["category"] == "Furniture") & (
    final_join["fraud_probability_consumer"] > 15.077) & (
    final_join["fraud_probability_merchant"]>30.941),1)\
    .when((final_join["category"] == "Beauty, Health, Personal and Household") 
    & (
    final_join["fraud_probability_consumer"] > 15.392) & (
    final_join["fraud_probability_merchant"]>29.76),1)\
    .when((final_join["category"] == "Electronics") & (
    final_join["fraud_probability_consumer"] > 15.085) &(
    final_join["fraud_probability_merchant"]>29.635),1)\
    .when((final_join["category"] == "Books, Stationary and Music") & (
    final_join["fraud_probability_consumer"] > 15.140) &(
    final_join["fraud_probability_merchant"]>29.016),1).otherwise(0)

# joining the boolean column to the rest of the data
model_with_fraud = final_join.withColumn("is_fraud",is_fraud)
final_model_with_fraud = model_with_fraud.groupBy("merchant_abn").agg(
    F.avg("is_fraud").alias("average_fraud_rate_per_merchant"))

model_with_fraud = model_with_fraud.drop("_c0", "merchant_name1", 
"merchant_abn1", 'categories')


#==============================================================================
# SAVING FINAL DATASETS
#==============================================================================

model_with_fraud.write.mode("overwrite").parquet(
    raw_internal_path + "transactions_with_fraud_rates.parquet")
final_model_with_fraud.write.mode("overwrite").parquet(
    raw_internal_path + "avg_fraud_rate_per_merchant.parquet")
#==============================================================================