#==============================================================================
# Import libraries
import pandas as pd
import sys, json
from pyspark.sql import SparkSession, functions as F
import lbl2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from pyspark.sql.functions import date_format
from pyspark.sql.functions import max
from pyspark.sql.functions import lit
from pyspark.sql.functions import year, month
from pyspark.sql.functions import col,isnan, when, count
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import year, month
from pyspark.sql.functions import col,isnan, when, count
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
#==============================================================================

# Create a spark session
spark = (
    SparkSession.builder.appName("MAST30034 Project 2 part 9")
    .config("spark.sql.repl.eagerEval.enabled", True) 
    .config("spark.sql.parquet.cacheMetadata", "true")
    .config("spark.sql.session.timeZone", "Etc/UTC")
    .config("spark.driver.memory", "6g")
    .config("spark.executor.memory", "10g")
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
# STEP 1: Prepare the main dataset
#==============================================================================
full_join = spark.read.parquet(curated_data_path + "full_join.parquet")
# Read the tagged model
tagged_merchants_sdf = spark.read.parquet(
    curated_data_path + "tagged_merchants.parquet")

# -----------------------------------------------------------------------------
# Rename the merchant column 
tagged_merchants_sdf = tagged_merchants_sdf.withColumnRenamed('merchant_abn',

    'tagged_merchant_abn'
)

# -----------------------------------------------------------------------------
# Join the final dataset to the tagged model
full_join.createOrReplaceTempView("join")
tagged_merchants_sdf.createOrReplaceTempView("tagged")

joint = spark.sql(""" 

SELECT *
FROM join
INNER JOIN tagged
ON join.merchant_abn = tagged.tagged_merchant_abn
""")

# -----------------------------------------------------------------------------
# Calculate the BNPL earnings 
joint = joint.drop('tagged_merchant_abn')
joint.createOrReplaceTempView("group")

main_data = spark.sql(""" 

SELECT *, ((take_rate/100)*dollar_value) AS BNPL_earnings
FROM group
""")


# -----------------------------------------------------------------------------
# Extracting the year, month, day from the timestamp
main_data = main_data.withColumn('Year', year(main_data.order_datetime))
main_data = main_data.withColumn('Month',month(main_data.order_datetime))

# -----------------------------------------------------------------------------
# Drop the unwanted columns
main_data = main_data.drop('merchant_abn', 'categories','name', 'address', 
'trans_merchant_abn', 'order_id','order_datetime','user_id','consumer_id',
'int_sa2', 'SA2_name','state_code','state_name','population_2020', 
'population_2021')

# -----------------------------------------------------------------------------
# Find Count of Null, None, NaN of All DataFrame Columns
null_values = main_data.select([count(when(isnan(c) | col(c).isNull(), 
c)).alias(c) for c in main_data.columns])

# -----------------------------------------------------------------------------
# Calculate the number of male and females customers for each merchant
main_data.createOrReplaceTempView("agg")

male = spark.sql(""" 

SELECT CONCAT(merchant_name, SA2_code, Year, Month) AS m_name, 
    COUNT(gender) as males
FROM agg
WHERE gender = 'Male'
GROUP BY merchant_name, SA2_code, Year, Month
""")

female = spark.sql(""" 

SELECT CONCAT(merchant_name, SA2_code, Year, Month) AS f_name, 
    COUNT(gender) as females
FROM agg
WHERE gender = 'Female'
GROUP BY merchant_name, SA2_code, Year, Month
""")

# -----------------------------------------------------------------------------
# Aggregate the main dataset
main_data.createOrReplaceTempView("agg")

main_agg = spark.sql(""" 

SELECT merchant_name, COUNT(merchant_name) AS no_of_transactions, SA2_code, 
Year, Month, SUM(BNPL_earnings) AS BNPL_earnings,
    CONCAT(merchant_name, SA2_code, Year, Month) AS join_col
FROM agg
GROUP BY merchant_name, SA2_code, Year, Month
""")

# -----------------------------------------------------------------------------
# Join the male and female customer counts to the main dataset
main_agg.createOrReplaceTempView("gender_join")
male.createOrReplaceTempView("m")
female.createOrReplaceTempView("f")

temp2 = spark.sql(""" 

SELECT *
FROM gender_join
INNER JOIN m
ON gender_join.join_col = m.m_name
""")

temp2.createOrReplaceTempView("temp2")

temp3 = spark.sql(""" 

SELECT *
FROM temp2
INNER JOIN f
ON temp2.join_col = f.f_name
""")


# -----------------------------------------------------------------------------
# Rename the column for better readability 
main_data = main_data.withColumnRenamed('income_2018-2019',

    'income_2018_2019'    
)

# -----------------------------------------------------------------------------
# Calculate the income per person for each SA2 code
main_data = main_data.withColumn('income_per_persons',
    (F.col('income_2018_2019')/F.col('total_persons'))
)

# -----------------------------------------------------------------------------
# Extract the revenue level and category for each merchant and total females,
# males and income per person for each SA2 code
main_data.createOrReplaceTempView("features")

e = spark.sql(""" 

SELECT merchant_name AS drop_name, FIRST(take_rate) AS take_rate, 
    FIRST(revenue_levels) AS revenue_levels, FIRST(category) AS category, 
    FIRST(total_males) AS males_in_SA2, FIRST(total_females) AS females_in_SA2,
    FIRST(income_per_persons) AS income_per_person
FROM features
GROUP BY merchant_name
""")

# -----------------------------------------------------------------------------
# Join the extracted values to the main dataset
temp3.createOrReplaceTempView("edit")
e.createOrReplaceTempView("rates")

temp4 = spark.sql(""" 

SELECT *
FROM edit
INNER JOIN rates
ON edit.merchant_name = rates.drop_name
""")

# -----------------------------------------------------------------------------
# Drop the redundant columns
train = temp4.drop('m_name', 'f_name', 'drop_name','join_col')
#==============================================================================
# STEP 2: Prepare a train and test dataset by offsetting the months by 1
#==============================================================================
# ----------------------------------------------------------------------------
# Select the main columns for offsetting
train_projection = train.select("merchant_name", "SA2_code", "Year", "Month", 
'BNPL_earnings')

# ----------------------------------------------------------------------------
# Offset the dataset by 1 month

# Offset the year by 1 if the month if the first month
train_projection = train_projection.withColumn("prev_year", \
              when(train_projection["Month"] == 1, 
              train_projection['Year'] - 1).otherwise(
                train_projection['Year']))

train_projection = train_projection.withColumn("prev_month", \
              when(train_projection["Month"] == 1, 
              12).otherwise(train_projection['Month'] - 1))

# Drop the redundant columns
train_projection = train_projection.drop("Year", "Month")

# Renam the columns
train_projection = train_projection.withColumnRenamed("BNPL_earnings", 
                "future_earnings") \
                .withColumnRenamed("merchant_name", "p_merchant_name") \
                .withColumnRenamed("SA2_code", "p_SA2_code")

# -----------------------------------------------------------------------------
# Join the offsetted values to the rest of the SA2 and aggregated values
final_data = train.join(train_projection, 
            (train.merchant_name == train_projection.p_merchant_name) & 
            (train.SA2_code == train_projection.p_SA2_code) & 
            (train.Year == train_projection.prev_year) & 
            (train.Month == train_projection.prev_month), how = 'inner')

# -----------------------------------------------------------------------------
# Drop the redundant columns
final_data = final_data.drop("p_merchant_name", "p_SA2_code","prev_year", 
"prev_month")

# ----------------------------------------------------------------------------- 
# Change the variable types
field_str = ['Year', 'Month', 'SA2_code']

for cols in field_str:
    final_data = final_data.withColumn(cols,F.col(cols).cast('STRING'))

field_int = ['no_of_transactions', 'males', 'females', 'males_in_SA2', 
'females_in_SA2']

for col in field_int:
    final_data = final_data.withColumn(col, F.col(col).cast('INT'))


#==============================================================================
# STEP 3: Build and train the Random Forrest Model
#==============================================================================
# String indexing the categorical columns

indexer = StringIndexer(inputCols = ['merchant_name', 'SA2_code', 'Year', 
'Month', 'revenue_levels','category'],
outputCols = ['merchant_name_num', 'SA2_code_num', 'Year_num', 'Month_num', 
'revenue_levels_num','category_num'], handleInvalid="keep")

indexd_data = indexer.fit(final_data).transform(final_data)


# Applying onehot encoding to the categorical data that is string indexed above
encoder = OneHotEncoder(inputCols = ['merchant_name_num', 'SA2_code_num', 
'Year_num', 'Month_num', 'revenue_levels_num','category_num'],
outputCols = ['merchant_name_vec', 'SA2_code_vec', 'Year_vec', 'Month_vec', 
'revenue_levels_vec','category_vec'])

onehotdata = encoder.fit(indexd_data).transform(indexd_data)


# Assembling the training data as a vector of features 
assembler1 = VectorAssembler(
inputCols=['merchant_name_vec', 'SA2_code_vec', 'Year_vec', 'Month_vec', 
'revenue_levels_vec','category_vec','males_in_SA2','females_in_SA2', 
'income_per_person', 'no_of_transactions','take_rate', 'BNPL_earnings'],
outputCol= "features" )

outdata1 = assembler1.transform(onehotdata)

# ----------------------------------------------------------------------------- 
# Renaming the target column as label

outdata1 = outdata1.withColumnRenamed(
    "future_earnings",
    "label"
)

# ----------------------------------------------------------------------------- 
# Assembling the features as a feature vector 

featureIndexer =\
    VectorIndexer(inputCol="features", 
    outputCol="indexedFeatures").fit(outdata1)

outdata1 = featureIndexer.transform(outdata1)

# ----------------------------------------------------------------------------- 
# Split the data into training and validation sets (30% held out for testing)
trainingData, testData = outdata1.randomSplit([0.7, 0.3], seed = 20)


# ----------------------------------------------------------------------------- 
# Train a RandomForest model.
rf = RandomForestRegressor(featuresCol="indexedFeatures")

# Train model.  
model = rf.fit(trainingData)

# Make predictions.
predictions_validation = model.transform(testData)

# ----------------------------------------------------------------------------- 
# Evaluate the validation set 
predictions_validation.select("prediction", "label", "features")

# Select (prediction, true label) and compute test error
evaluator_train_rmse = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse_train = evaluator_train_rmse.evaluate(predictions_validation)

evaluator_train_mae = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="mae")
mae_train = evaluator_train_mae.evaluate(predictions_validation)

BNPL_metrics = {
    'RMSE': [rmse_train],
    'MAE': [mae_train]
}

BNPL_metrics_df = pd.DataFrame(BNPL_metrics)
BNPL_metrics_df.to_csv(curated_data_path + "BNPL_metrics.csv")

# ----------------------------------------------------------------------------- 
# Define a function to extract important feature column names
def ExtractFeatureImportance(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol
                    ].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))
  
# ----------------------------------------------------------------------------- 
#ExtractFeatureImportance(model.stages[-1].featureImportances, dataset, 
# "features")
dataset_fi = ExtractFeatureImportance(model.featureImportances, 
predictions_validation, "features")
dataset_fi = spark.createDataFrame(dataset_fi)
dataset_fi_BNPL_df = dataset_fi.toPandas()
dataset_fi_BNPL_df.to_csv(curated_data_path + "BNPL_features.csv")

# ----------------------------------------------------------------------------- 
# Select the latest month from the latest year in the dataset which will be
# used as a test set for future predictions due to the offsetting done 
# previously

latest_year = train.select(max('Year')).collect()[0][0]
agg_month_1 = train.filter(train.Year == latest_year)
latest_month = agg_month_1.select(max('Month')).collect()[0][0]
predicting_data = agg_month_1.filter(train.Month == latest_month)
predicting_data = predicting_data.withColumn("future_earnings", lit(0))
# Change the variable types
field_str = ['Year', 'Month', 'SA2_code']

for cols in field_str:
    predicting_data = predicting_data.withColumn(cols,
    F.col(cols).cast('STRING'))

field_int = ['no_of_transactions', 'males', 'females', 'males_in_SA2', 
'females_in_SA2']

for col in field_int:
    predicting_data = predicting_data.withColumn(col, F.col(col).cast('INT'))


#==============================================================================
# STEP 4: Make future predictions
#==============================================================================
# Repeat the indexing and vector assembling steps again for the test data
# String indexing the categorical columns

indexer = StringIndexer(inputCols = ['merchant_name', 'SA2_code', 'Year', 
'Month', 'revenue_levels','category'],
outputCols = ['merchant_name_num', 'SA2_code_num', 'Year_num', 'Month_num', 
'revenue_levels_num','category_num'], handleInvalid="keep")

indexd_data = indexer.fit(predicting_data).transform(predicting_data)

# Applying onehot encoding to the categorical data that is string indexed above
encoder = OneHotEncoder(inputCols = ['merchant_name_num', 'SA2_code_num', 
'Year_num', 'Month_num', 'revenue_levels_num','category_num'],
outputCols = ['merchant_name_vec', 'SA2_code_vec', 'Year_vec', 'Month_vec', 
'revenue_levels_vec','category_vec'])

onehotdata = encoder.fit(indexd_data).transform(indexd_data)

# Assembling the training data as a vector of features 
assembler1 = VectorAssembler(
inputCols=['merchant_name_vec', 'SA2_code_vec', 'Year_vec', 'Month_vec', 
'revenue_levels_vec','category_vec','males_in_SA2','females_in_SA2', 
'income_per_person', 'no_of_transactions','take_rate', 'BNPL_earnings'],
outputCol= "features" )

outdata1 = assembler1.transform(onehotdata)

# ----------------------------------------------------------------------------- 
# Renaming the target column as label

outdata1 = outdata1.withColumnRenamed(
    "future_earnings",
    "label"
)

# ----------------------------------------------------------------------------- 
# Assembling the features as a feature vector 

featureIndexer =\
    VectorIndexer(inputCol="features", 
    outputCol="indexedFeatures").fit(outdata1)

# ----------------------------------------------------------------------------- 
# Transform the test data
outdata1 = featureIndexer.transform(outdata1)

predictions_test = model.transform(outdata1)
# ----------------------------------------------------------------------------- 
# Aggregate the predictions to merchant level to get the predicted BNPL 
# earnings from each merchant
predictions_test.createOrReplaceTempView("preds")

pred = spark.sql(""" 

SELECT merchant_name, SUM(prediction) AS total_earnings_of_BNPL
FROM preds
GROUP BY merchant_name

""")

# -----------------------------------------------------------------------------  
# Convert the predictions to a pandas dataframe and save as a csv
pred_df = pred.toPandas()

pred_df.to_csv(curated_data_path + "BNPL_earnings.csv")
# ----------------------------------------------------------------------------- 