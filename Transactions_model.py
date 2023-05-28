import pandas as pd
import sys, json
from pyspark.sql import SparkSession, functions as F
#==============================================================================
import lbl2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from pyspark.sql.functions import max, lit
from pyspark.sql.functions import date_format
import statsmodels.api as sm
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col,isnan, when, count
from pyspark.sql.functions import year, month
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

#==============================================================================
# Create a spark session
spark = (
    SparkSession.builder.appName("MAST30034 Project 2 part 8")
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
    curated_data_path + "tagged_merchants.parquet"
    )

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

a_transactions = spark.sql(""" 

SELECT *, (take_rate/100)*dollar_value AS percent
FROM group
""")

# -----------------------------------------------------------------------------
# Extracting the year, month, day from the timestamp
a_transactions = a_transactions.withColumn('Year', 
year(a_transactions.order_datetime))
a_transactions = a_transactions.withColumn('Month',
month(a_transactions.order_datetime))

# -----------------------------------------------------------------------------
# Drop the unwanted columns
a_transactions = a_transactions.drop('merchant_abn','categories','name',
'address','trans_merchant_abn', 'user_id','order_id','order_datetime', 
'consumer_id', 'int_sa2', 'SA2_name','state_code', 'state_name',
'population_2020', 'population_2021')
 

# -----------------------------------------------------------------------------
# Find Count of Null, None, NaN of All DataFrame Columns
a_transactions.select([count(when(isnan(c) | col(c).isNull(), 
c)).alias(c) for c in a_transactions.columns])

# -----------------------------------------------------------------------------
# Calculate the number of male and females transactionss for each merchant
a_transactions.createOrReplaceTempView("agg")

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
a_transactions.createOrReplaceTempView("agg")

temp_transactions = spark.sql(""" 

SELECT merchant_name, COUNT(merchant_name) AS no_of_transactions, SA2_code, 
Year, Month, SUM(dollar_value - percent) AS total_earnings,
    CONCAT(merchant_name, SA2_code, Year, Month) AS join_col
FROM agg 
GROUP BY merchant_name, SA2_code, Year, Month
""")

# -----------------------------------------------------------------------------
# Join the male and female transactions counts to the main dataset
temp_transactions.createOrReplaceTempView("gender_join")
male.createOrReplaceTempView("m")
female.createOrReplaceTempView("f")

temp2_transactions = spark.sql(""" 

SELECT *
FROM gender_join
INNER JOIN m
ON gender_join.join_col = m.m_name
""")

temp2_transactions.createOrReplaceTempView("temp2")

temp3_transactions = spark.sql(""" 

SELECT *
FROM temp2
INNER JOIN f
ON temp2.join_col = f.f_name
""")


# -----------------------------------------------------------------------------
# Rename the column for better readability 
a_transactions = a_transactions.withColumnRenamed('income_2018-2019',

    'income_2018_2019'    
)

# -----------------------------------------------------------------------------
# Calculate the income per person for each SA2 code
a_transactions = a_transactions.withColumn('income_per_persons',
    (F.col('income_2018_2019')/F.col('total_persons'))
)

# -----------------------------------------------------------------------------
# Extract the revenue level and category for each merchant and total females,
# males and income per person for each SA2 code
a_transactions.createOrReplaceTempView("features")

e_transactions = spark.sql(""" 

SELECT merchant_name AS drop_name, FIRST(take_rate) AS take_rate, 
    FIRST(revenue_levels) AS revenue_levels, FIRST(category) AS category,
    FIRST(total_males) AS males_in_SA2, FIRST(total_females) AS females_in_SA2, 
    FIRST(income_per_persons) AS income_per_person
FROM features
GROUP BY merchant_name
""")

# -----------------------------------------------------------------------------
# Join the extracted values to the main dataset
temp3_transactions.createOrReplaceTempView("edit")
e_transactions.createOrReplaceTempView("rates")

temp4_transactions = spark.sql(""" 

SELECT *
FROM edit
INNER JOIN rates
ON edit.merchant_name = rates.drop_name
""")

# -----------------------------------------------------------------------------
# Drop the redundant columns
train_transactions = temp4_transactions.drop('m_name', 'f_name', 'drop_name',
'join_col')

#==============================================================================
# STEP 2: Prepare a train and test dataset by offsetting the months by 1
#==============================================================================
# ----------------------------------------------------------------------------
# Select the main columns for offsetting
train_projection_transactions = train_transactions.select("merchant_name", 
"SA2_code", "Year", "Month", 'no_of_transactions')

# ----------------------------------------------------------------------------
# Offset the dataset by 1 month

# Offset the year by 1 if the month if the first month
train_projection_transactions = train_projection_transactions.withColumn(
    "prev_year",\
              when(train_projection_transactions["Month"] == 1, 
              train_projection_transactions['Year'
              ] - 1).otherwise(train_projection_transactions['Year']))
train_projection_transactions = train_projection_transactions.withColumn(
    "prev_month",\
              when(train_projection_transactions["Month"] == 1, 12
              ).otherwise(train_projection_transactions['Month'] - 1))

# Drop the redundant columns
train_projection_transactions = train_projection_transactions.drop("Year", 
"Month")

# Renam the columns
train_projection_transactions = train_projection_transactions.withColumnRenamed(
            "no_of_transactions", 
            "future_transactions") \
            .withColumnRenamed("merchant_name", 
            "p_merchant_name") \
            .withColumnRenamed("SA2_code", "p_SA2_code")

# -----------------------------------------------------------------------------
# Join the offsetted values to the rest of the SA2 and aggregated values
final_data_transactions = train_transactions.join(train_projection_transactions, 
(train_transactions.merchant_name == train_projection_transactions.p_merchant_name) & 
(train_transactions.SA2_code == train_projection_transactions.p_SA2_code) & 
(train_transactions.Year == train_projection_transactions.prev_year) & 
(train_transactions.Month == train_projection_transactions.prev_month),
 how = 'inner')

# -----------------------------------------------------------------------------
# Drop the redundant columns
final_data_transactions = final_data_transactions.drop("p_merchant_name", 
"p_SA2_code","prev_year", "prev_month")

# ----------------------------------------------------------------------------- 
# Change the variable types
field_str_transactions = ['Year', 'Month', 'SA2_code']

for cols in field_str_transactions:
    final_data_transactions = final_data_transactions.withColumn(cols,
    F.col(cols).cast('STRING'))

field_int_transactions = ['no_of_transactions', 'males', 'females', 
'males_in_SA2', 'females_in_SA2']

for col in field_int_transactions:
    final_data_transactions = final_data_transactions.withColumn(col, 
    F.col(col).cast('INT'))

#==============================================================================
# STEP 3: Build and train the Random Forrest Model
#==============================================================================
# String indexing the categorical columns

field = ['future_transactions','no_of_transactions' ,'males', 'females', 
        'males_in_SA2', 'females_in_SA2']

for col in field:
    final_data_transactions = final_data_transactions.withColumn(col,

    F.col(col).cast('INT')

)
# String indexing the categorical columns

indexer_transactions = StringIndexer(inputCols = ['merchant_name', 'SA2_code', 
'Year', 'Month', 'revenue_levels','category'],
outputCols = ['merchant_name_num', 'SA2_code_num', 'Year_num', 'Month_num', 
'revenue_levels_num','category_num'], handleInvalid="keep")

indexd_data_transactions = indexer_transactions.fit(final_data_transactions
).transform(final_data_transactions)


# Applying onehot encoding to the categorical data that is string indexed above
encoder_transactions = OneHotEncoder(inputCols = ['merchant_name_num', 
'SA2_code_num', 'Year_num', 'Month_num', 'revenue_levels_num','category_num'],
outputCols = ['merchant_name_vec', 'SA2_code_vec', 'Year_vec', 'Month_vec', 
'revenue_levels_vec','category_vec'])

onehotdata_transactions = encoder_transactions.fit(indexd_data_transactions
).transform(indexd_data_transactions)


# Assembling the training data as a vector of features 
assembler1_transactions = VectorAssembler(
inputCols=['merchant_name_vec', 'SA2_code_vec', 'Year_vec', 'Month_vec', 
'revenue_levels_vec','category_vec','males_in_SA2','females_in_SA2',
'no_of_transactions' ,'income_per_person','take_rate', 'total_earnings'],
outputCol= "features" )

outdata1_transactions = assembler1_transactions.transform(
                                    onehotdata_transactions)

# ----------------------------------------------------------------------------- 
# Renaming the target column as label

outdata1_transactions = outdata1_transactions.withColumnRenamed(
    "future_transactions",
    "label"
)

# ----------------------------------------------------------------------------- 
# Assembling the features as a feature vector 
featureIndexer_transactions =\
    VectorIndexer(inputCol="features", 
    outputCol="indexedFeatures").fit(outdata1_transactions)

outdata1_transactions = featureIndexer_transactions.transform(outdata1_transactions)


# ----------------------------------------------------------------------------- 
# Split the data into training and validation sets (30% held out for testing)

trainingData_transactions, testData_transactions = outdata1_transactions.randomSplit([0.7,
0.3], seed = 20)

# ----------------------------------------------------------------------------- 
# Train a RandomForest model.
rf = RandomForestRegressor(featuresCol="indexedFeatures")

# ----------------------------------------------------------------------------- 
# Train model.  
model_transactions = rf.fit(trainingData_transactions)

# ----------------------------------------------------------------------------- 
# Make predictions.
predictions_validation_transactions = model_transactions.transform(testData_transactions)

# ----------------------------------------------------------------------------- 
# Evaluate the validation set 

predictions_validation_transactions.select("prediction", "label", "features")

# ----------------------------------------------------------------------------- 
# Select (prediction, true label) and compute test error

evaluator_train_rmse_transactions = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse_train_transactions = evaluator_train_rmse_transactions.evaluate(
    predictions_validation_transactions)

evaluator_train_mae_transactions = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="mae")
mae_train_transactions = evaluator_train_mae_transactions.evaluate(
    predictions_validation_transactions)

transactions_metrics = {
    'RMSE': [rmse_train_transactions],
    'MAE': [mae_train_transactions]
}

transactions_metrics_df = pd.DataFrame(transactions_metrics)
transactions_metrics_df.to_csv(curated_data_path + "transactions_metrics.csv")
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
  
  # ---------------------------------------------------------------------------
#ExtractFeatureImportance(model.stages[-1].featureImportances, dataset, 
# "features")
dataset_fi_transactions = ExtractFeatureImportance(
    model_transactions.featureImportances, 
predictions_validation_transactions, "features")
dataset_fi_transactions = spark.createDataFrame(dataset_fi_transactions)

dataset_fi_transactions_df = dataset_fi_transactions.toPandas()
dataset_fi_transactions_df.to_csv(
    curated_data_path + "transactions_features.csv")
# ----------------------------------------------------------------------------- 
# Select the latest month from the latest year in the dataset which will be
# used as a test set for future predictions due to the offsetting done 
# previously

latest_year_transactions = train_transactions.select(max('Year')).collect()[0][0]
agg_month_1_transactions = train_transactions.filter(
    train_transactions.Year == latest_year_transactions)
latest_month_transactions = agg_month_1_transactions.select(
    max('Month')).collect()[0][0]
predicting_data_transactions = agg_month_1_transactions.filter(
    train_transactions.Month == latest_month_transactions)
predicting_data_transactions = predicting_data_transactions.withColumn(
    "future_transactionss", lit(0))

# ----------------------------------------------------------------------------- 
# Change the variable types
field_str_transactions = ['Year', 'Month', 'SA2_code']

for cols in field_str_transactions:
    predicting_data_transactions = predicting_data_transactions.withColumn(cols,
    F.col(cols).cast('STRING'))

field_int_transactions = ['no_of_transactions', 'males', 'females', 
'males_in_SA2', 'females_in_SA2']

for col in field_int_transactions:
    predicting_data_transactions = predicting_data_transactions.withColumn(col, 
    F.col(col).cast('INT'))
#==============================================================================
# STEP 4: Make future predictions
#==============================================================================
# Repeat the indexing and vector assembling steps again for the test data
# String indexing the categorical columns
indexer_transactions = StringIndexer(inputCols = ['merchant_name', 'SA2_code', 
'Year', 'Month', 'revenue_levels','category'],
outputCols = ['merchant_name_num', 'SA2_code_num', 'Year_num', 'Month_num', 
'revenue_levels_num','category_num'], handleInvalid="keep")

indexd_data_transactions = indexer_transactions.fit(predicting_data_transactions
).transform(predicting_data_transactions)


# Applying onehot encoding to the categorical data that is string indexed above
encoder_transactions = OneHotEncoder(inputCols = ['merchant_name_num', 
'SA2_code_num','Year_num', 'Month_num', 'revenue_levels_num','category_num'],
outputCols = ['merchant_name_vec', 'SA2_code_vec', 'Year_vec', 'Month_vec', 
'revenue_levels_vec','category_vec'])

onehotdata_transactions = encoder_transactions.fit(indexd_data_transactions
).transform(indexd_data_transactions)


# Assembling the training data as a vector of features 
assembler1_transactions = VectorAssembler(
inputCols=['merchant_name_vec', 'SA2_code_vec', 'Year_vec', 'Month_vec', 
'revenue_levels_vec','category_vec','males_in_SA2','females_in_SA2', 
'income_per_person', 'no_of_transactions','take_rate', 'total_earnings'],
outputCol= "features" )

outdata1_transactions = assembler1_transactions.transform(
                                    onehotdata_transactions)

# ----------------------------------------------------------------------------- 
# Renaming the target column as label

outdata1_transactions = outdata1_transactions.withColumnRenamed(
    "future_transactionss",
    "label"
)

# ----------------------------------------------------------------------------- 
# Assembling the features as a feature vector 
featureIndexer_transactions =\
    VectorIndexer(inputCol="features", 
    outputCol="indexedFeatures").fit(outdata1_transactions)


# ----------------------------------------------------------------------------- 
# Transform the test data
outdata1_transactions = featureIndexer_transactions.transform(
                                        outdata1_transactions)
predictions_test_transactions = model_transactions.transform(
                                        outdata1_transactions)

# ----------------------------------------------------------------------------- 
# Aggregate the predictions to merchant level to get the predicted BNPL 
# earnings from each merchant
predictions_test_transactions.createOrReplaceTempView("preds")

pred_transactions = spark.sql(""" 

SELECT merchant_name, ROUND(SUM(prediction)) AS total_future_transactionss
FROM preds
GROUP BY merchant_name

""")

# -----------------------------------------------------------------------------  
# Convert the predictions to a pandas dataframe and save as a csv
pred_df_transactions = pred_transactions.toPandas()
pred_df_transactions.to_csv(curated_data_path + "transactionss.csv")
# ----------------------------------------------------------------------------- 