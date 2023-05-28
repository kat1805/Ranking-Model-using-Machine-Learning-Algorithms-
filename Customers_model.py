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

a_customer = spark.sql(""" 

SELECT *, (take_rate/100)*dollar_value AS percent
FROM group
""")

# -----------------------------------------------------------------------------
# Extracting the year, month, day from the timestamp
a_customer = a_customer.withColumn('Year', year(a_customer.order_datetime))
a_customer = a_customer.withColumn('Month',month(a_customer.order_datetime))

# -----------------------------------------------------------------------------
# Drop the unwanted columns
a_customer = a_customer.drop('merchant_abn', 'categories','name', 'address', 
'trans_merchant_abn', 'order_id','order_datetime', 'consumer_id','int_sa2', 
'SA2_name','state_code','state_name','population_2020', 'population_2021')
 

# -----------------------------------------------------------------------------
# Find Count of Null, None, NaN of All DataFrame Columns
a_customer.select([count(when(isnan(c) | col(c).isNull(), 
c)).alias(c) for c in a_customer.columns])

# -----------------------------------------------------------------------------
# Calculate the number of male and females customers for each merchant
a_customer.createOrReplaceTempView("agg")

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
a_customer.createOrReplaceTempView("agg")

temp_customer = spark.sql(""" 

SELECT merchant_name, COUNT(DISTINCT user_id) AS no_of_customers, SA2_code, 
Year, Month, SUM(dollar_value - percent) AS total_earnings,
    CONCAT(merchant_name, SA2_code, Year, Month) AS join_col
FROM agg 
GROUP BY merchant_name, SA2_code, Year, Month
""")

# -----------------------------------------------------------------------------
# Join the male and female customer counts to the main dataset
temp_customer.createOrReplaceTempView("gender_join")
male.createOrReplaceTempView("m")
female.createOrReplaceTempView("f")

temp2_customer = spark.sql(""" 

SELECT *
FROM gender_join
INNER JOIN m
ON gender_join.join_col = m.m_name
""")

temp2_customer.createOrReplaceTempView("temp2")

temp3_customer = spark.sql(""" 

SELECT *
FROM temp2
INNER JOIN f
ON temp2.join_col = f.f_name
""")


# -----------------------------------------------------------------------------
# Rename the column for better readability 
a_customer = a_customer.withColumnRenamed('income_2018-2019',

    'income_2018_2019'    
)

# -----------------------------------------------------------------------------
# Calculate the income per person for each SA2 code
a_customer = a_customer.withColumn('income_per_persons',
    (F.col('income_2018_2019')/F.col('total_persons'))
)

# -----------------------------------------------------------------------------
# Extract the revenue level and category for each merchant and total females,
# males and income per person for each SA2 code
a_customer.createOrReplaceTempView("features")

e_customer = spark.sql(""" 

SELECT merchant_name AS drop_name, FIRST(take_rate) AS take_rate, 
    FIRST(revenue_levels) AS revenue_levels, FIRST(category) AS category,
    FIRST(total_males) AS males_in_SA2, FIRST(total_females) AS females_in_SA2, 
    FIRST(income_per_persons) AS income_per_person
FROM features
GROUP BY merchant_name
""")

# -----------------------------------------------------------------------------
# Join the extracted values to the main dataset
temp3_customer.createOrReplaceTempView("edit")
e_customer.createOrReplaceTempView("rates")

temp4_customer = spark.sql(""" 

SELECT *
FROM edit
INNER JOIN rates
ON edit.merchant_name = rates.drop_name
""")

# -----------------------------------------------------------------------------
# Drop the redundant columns
train_customer = temp4_customer.drop('m_name', 'f_name', 'drop_name','join_col')

#==============================================================================
# STEP 2: Prepare a train and test dataset by offsetting the months by 1
#==============================================================================
# ----------------------------------------------------------------------------
# Select the main columns for offsetting
train_projection_customer = train_customer.select("merchant_name", "SA2_code", 
"Year", "Month", 'no_of_customers')

# ----------------------------------------------------------------------------
# Offset the dataset by 1 month

# Offset the year by 1 if the month if the first month
train_projection_customer = train_projection_customer.withColumn("prev_year", \
              when(train_projection_customer["Month"] == 1, 
              train_projection_customer['Year'] - 1).otherwise(
                train_projection_customer['Year']))
train_projection_customer = train_projection_customer.withColumn("prev_month", \
              when(train_projection_customer["Month"] == 1, 12
              ).otherwise(train_projection_customer['Month'] - 1))

# Drop the redundant columns
train_projection_customer = train_projection_customer.drop("Year", "Month")

# Renam the columns
train_projection_customer = train_projection_customer.withColumnRenamed(
            "no_of_customers", 
            "future_customers") \
            .withColumnRenamed("merchant_name", 
            "p_merchant_name") \
            .withColumnRenamed("SA2_code", "p_SA2_code")

# -----------------------------------------------------------------------------
# Join the offsetted values to the rest of the SA2 and aggregated values
final_data_customer = train_customer.join(train_projection_customer, 
(train_customer.merchant_name == train_projection_customer.p_merchant_name) & 
(train_customer.SA2_code == train_projection_customer.p_SA2_code) & 
(train_customer.Year == train_projection_customer.prev_year) & 
(train_customer.Month == train_projection_customer.prev_month), how = 'inner')

# -----------------------------------------------------------------------------
# Drop the redundant columns
final_data_customer = final_data_customer.drop("p_merchant_name", "p_SA2_code",
"prev_year", "prev_month")

# ----------------------------------------------------------------------------- 
# Change the variable types
field_str_customer = ['Year', 'Month', 'SA2_code']

for cols in field_str_customer:
    final_data_customer = final_data_customer.withColumn(cols,F.col(cols
    ).cast('STRING'))

field_int_customer = ['no_of_customers', 'males', 'females', 'males_in_SA2', 
'females_in_SA2']

for col in field_int_customer:
    final_data_customer = final_data_customer.withColumn(col, 
    F.col(col).cast('INT'))

#==============================================================================
# STEP 3: Build and train the Random Forrest Model
#==============================================================================
# String indexing the categorical columns

field = ['future_customers','no_of_customers' ,'males', 'females', 
        'males_in_SA2', 'females_in_SA2']

for col in field:
    final_data_customer = final_data_customer.withColumn(col,

    F.col(col).cast('INT')

)
# String indexing the categorical columns

indexer_customer = StringIndexer(inputCols = ['merchant_name', 'SA2_code', 
'Year', 'Month', 'revenue_levels','category'],
outputCols = ['merchant_name_num', 'SA2_code_num', 'Year_num', 'Month_num', 
'revenue_levels_num','category_num'], handleInvalid="keep")

indexd_data_customer = indexer_customer.fit(final_data_customer
).transform(final_data_customer)


# Applying onehot encoding to the categorical data that is string indexed above
encoder_customer = OneHotEncoder(inputCols = ['merchant_name_num', 
'SA2_code_num', 'Year_num', 'Month_num', 'revenue_levels_num','category_num'],
outputCols = ['merchant_name_vec', 'SA2_code_vec', 'Year_vec', 'Month_vec', 
'revenue_levels_vec','category_vec'])

onehotdata_customer = encoder_customer.fit(indexd_data_customer
).transform(indexd_data_customer)


# Assembling the training data as a vector of features 
assembler1_customer = VectorAssembler(
inputCols=['merchant_name_vec', 'SA2_code_vec', 'Year_vec', 'Month_vec', 
'revenue_levels_vec','category_vec','males_in_SA2','females_in_SA2',
'no_of_customers' ,'income_per_person','take_rate', 'total_earnings'],
outputCol= "features" )

outdata1_customer = assembler1_customer.transform(onehotdata_customer)
# Renaming the target column as label

# ----------------------------------------------------------------------------- 
# Renaming the target column as label

outdata1_customer = outdata1_customer.withColumnRenamed(
    "future_customers",
    "label"
)

# ----------------------------------------------------------------------------- 
# Assembling the features as a feature vector 
featureIndexer_customer =\
    VectorIndexer(inputCol="features", 
    outputCol="indexedFeatures").fit(outdata1_customer)

outdata1_customer = featureIndexer_customer.transform(outdata1_customer)


# ----------------------------------------------------------------------------- 
# Split the data into training and validation sets (30% held out for testing)

trainingData_customer, testData_customer = outdata1_customer.randomSplit(
    [0.7, 0.3], seed = 20)

# ----------------------------------------------------------------------------- 
# Train a RandomForest model.
rf = RandomForestRegressor(featuresCol="indexedFeatures")

# ----------------------------------------------------------------------------- 
# Train model.  
model_customer = rf.fit(trainingData_customer)

# ----------------------------------------------------------------------------- 
# Make predictions.
predictions_validation_customer = model_customer.transform(testData_customer)

# ----------------------------------------------------------------------------- 
# Evaluate the validation set 

predictions_validation_customer.select("prediction", "label", "features")

# ----------------------------------------------------------------------------- 
# Select (prediction, true label) and compute test error

evaluator_train_rmse_customer = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse_train_customer = evaluator_train_rmse_customer.evaluate(
    predictions_validation_customer)

evaluator_train_mae_customer = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="mae")
mae_train_customer = evaluator_train_mae_customer.evaluate(
    predictions_validation_customer)

customer_metrics = {
    'RMSE': [rmse_train_customer],
    'MAE': [mae_train_customer]
}

customer_metrics_df = pd.DataFrame(customer_metrics)
customer_metrics_df.to_csv(curated_data_path + "customer_metrics.csv")
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
dataset_fi_customer = ExtractFeatureImportance(
    model_customer.featureImportances, 
predictions_validation_customer, "features")
dataset_fi_customer = spark.createDataFrame(dataset_fi_customer)

dataset_fi_customer_df = dataset_fi_customer.toPandas()
dataset_fi_customer_df.to_csv(curated_data_path + "customer_features.csv")
# ----------------------------------------------------------------------------- 
# Select the latest month from the latest year in the dataset which will be
# used as a test set for future predictions due to the offsetting done 
# previously

latest_year_customer = train_customer.select(max('Year')).collect()[0][0]
agg_month_1_customer = train_customer.filter(
    train_customer.Year == latest_year_customer)
latest_month_customer = agg_month_1_customer.select(max('Month')
).collect()[0][0]
predicting_data_customer = agg_month_1_customer.filter(
    train_customer.Month == latest_month_customer)
predicting_data_customer = predicting_data_customer.withColumn(
    "future_customers", lit(0))

# ----------------------------------------------------------------------------- 
# Change the variable types
field_str_customer = ['Year', 'Month', 'SA2_code']

for cols in field_str_customer:
    predicting_data_customer = predicting_data_customer.withColumn(
        cols,F.col(cols).cast('STRING'))

field_int_customer = ['no_of_customers', 'males', 'females', 'males_in_SA2', 
'females_in_SA2']

for col in field_int_customer:
    predicting_data_customer = predicting_data_customer.withColumn(col, 
    F.col(col).cast('INT'))
#==============================================================================
# STEP 4: Make future predictions
#==============================================================================
# Repeat the indexing and vector assembling steps again for the test data
# String indexing the categorical columns
indexer_customer = StringIndexer(inputCols = ['merchant_name', 'SA2_code','Year', 
'Month', 'revenue_levels','category'],
outputCols = ['merchant_name_num', 'SA2_code_num', 'Year_num', 'Month_num', 
'revenue_levels_num','category_num'], handleInvalid="keep")

indexd_data_customer = indexer_customer.fit(predicting_data_customer
).transform(predicting_data_customer)


# Applying onehot encoding to the categorical data that is string indexed above
encoder_customer = OneHotEncoder(inputCols = ['merchant_name_num','SA2_code_num', 
'Year_num', 'Month_num', 'revenue_levels_num','category_num'],
outputCols = ['merchant_name_vec', 'SA2_code_vec', 'Year_vec', 'Month_vec', 
'revenue_levels_vec','category_vec'])

onehotdata_customer = encoder_customer.fit(indexd_data_customer
).transform(indexd_data_customer)


# Assembling the training data as a vector of features 
assembler1_customer = VectorAssembler(
inputCols=['merchant_name_vec', 'SA2_code_vec', 'Year_vec', 'Month_vec', 
'revenue_levels_vec','category_vec','males_in_SA2','females_in_SA2', 
'income_per_person', 'no_of_customers','take_rate', 'total_earnings'],
outputCol= "features" )

outdata1_customer = assembler1_customer.transform(onehotdata_customer)

# ----------------------------------------------------------------------------- 
# Renaming the target column as label

outdata1_customer = outdata1_customer.withColumnRenamed(
    "future_customers",
    "label"
)

# ----------------------------------------------------------------------------- 
# Assembling the features as a feature vector 
featureIndexer_customer =\
    VectorIndexer(inputCol="features", 
    outputCol="indexedFeatures").fit(outdata1_customer)


# ----------------------------------------------------------------------------- 
# Transform the test data
outdata1_customer = featureIndexer_customer.transform(outdata1_customer)
predictions_test_customer = model_customer.transform(outdata1_customer)

# ----------------------------------------------------------------------------- 
# Aggregate the predictions to merchant level to get the predicted BNPL 
# earnings from each merchant
predictions_test_customer.createOrReplaceTempView("preds")

pred_customer = spark.sql(""" 

SELECT merchant_name, ROUND(SUM(prediction)) AS total_future_customers
FROM preds
GROUP BY merchant_name

""")

# -----------------------------------------------------------------------------  
# Convert the predictions to a pandas dataframe and save as a csv
pred_df_customer = pred_customer.toPandas()
pred_df_customer.to_csv(curated_data_path + "customers.csv")
# ----------------------------------------------------------------------------- 