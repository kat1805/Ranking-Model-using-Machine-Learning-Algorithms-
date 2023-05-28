#==============================================================================
# Importing required libraries
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F
from pyspark.sql.functions import *
import json
from urllib.request import urlretrieve
from zipfile import ZipFile
import os
import sys
import geopandas as gpd

#==============================================================================
# Create a spark session
#==============================================================================
spark = (
    SparkSession.builder.appName("MAST30034 Project 2 part 2")
    .config("spark.sql.repl.eagerEval.enabled", True) 
    .config("spark.sql.parquet.cacheMetadata", "true")
    .config("spark.sql.session.timeZone", "Etc/UTC")
    .config("spark.driver.memory", "15g")
    .config("spark.executor.memory", "5g")
    .getOrCreate()
)

#==============================================================================
# EXTRACT 
#==============================================================================
# EXTRACT INTERNAL DATA FROM TABLES DIRECTORY
#==============================================================================
# Define relative target directories

paths_arg = sys.argv[1]

with open(paths_arg) as json_paths: 
    PATHS = json.load(json_paths)
    json_paths.close()

raw_internal_path = PATHS['raw_internal_data_path']
curated_data_path = PATHS['curated_data_path']
external_data_path = PATHS['external_data_path']
#------------------------------------------------------------------------------
    
# TBL Consumer
tbl_consumer = spark.read.option("header", True).csv(
    raw_internal_path +'tbl_consumer.csv', sep='|')

#--------------------------------------------------------------------------------------------
# TBL Merchants
tbl_merchants = spark.read.parquet(raw_internal_path + 'tbl_merchants.parquet')

#------------------------------------------------------------------------------
# Consumer User Details
user_details = spark.read.parquet(
    raw_internal_path + 'consumer_user_details.parquet')

#------------------------------------------------------------------------------
# Transactions
transactions1 = spark.read.parquet(
    raw_internal_path + 'transactions_20210228_20210827_snapshot/')
transactions2 = spark.read.parquet(
    raw_internal_path + 'transactions_20210828_20220227_snapshot/')
transactions3 = spark.read.parquet(
    raw_internal_path + 'transactions_20220228_20220828_snapshot/')
transactions12 = transactions1.union(
    transactions2).distinct()
transactions = transactions12.union(
    transactions3).distinct()

#------------------------------------------------------------------------------
# Fraud Details
fraud_consumer = spark.read.option("header", True).csv(
    raw_internal_path +'consumer_fraud_probability.csv')
fraud_merchants = spark.read.option("header", True).csv(
    raw_internal_path +'merchant_fraud_probability.csv')

#------------------------------------------------------------------------------
# Extract time periods (years) from transactions dataset
transactions = transactions.orderBy("order_datetime")

first_transaction_date = transactions.select(first("order_datetime"
).alias('date'))
first_transaction_year = first_transaction_date.withColumn("year", 
year(col('date')))

last_transaction_date = transactions.select(last("order_datetime"
).alias('date'))
last_transaction_year = last_transaction_date.withColumn("year", 
year(col('date')))

start_year = first_transaction_year.head()[1]
end_year = last_transaction_year.head()[1]

useful_years = list(range(start_year, end_year+1))

#==============================================================================
# EXTRACT EXTERNAL DATA
#==============================================================================
# Download Covid Data
#==============================================================================
# Specify the url
url = "https://raw.githubusercontent.com/M3IT/COVID-19_Data/master/Data/COVID_AU_state_daily_change.csv"

#------------------------------------------------------------------------------
# Define the file names
output_csv = external_data_path + "covid.csv"

#------------------------------------------------------------------------------
# Download the data
urlretrieve(url, output_csv) 

#==============================================================================
# MAKE NEW DIRECTORIES TO SAVE THE ABS DATASETS
#==============================================================================
# check if it exists as it makedir will raise an error if it does exist
if not os.path.exists(external_data_path):
    os.makedirs(external_data_path)

#------------------------------------------------------------------------------
# Define the directory names
dirs = ['SA2_boundaries', 'SA2_total_population', 'SA2_income', 'SA2_census']

#------------------------------------------------------------------------------
# now, for each type of data set we will need, we will create the paths
for target_dir in dirs: # taxi_zones should already exist
    if not os.path.exists(external_data_path + target_dir):
        os.makedirs(external_data_path + target_dir)

#==============================================================================
# SA2 BOUNDARIES DATASET DOWNLOAD
#==============================================================================

# Specify the url
SA2_URL_ZIP = "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA2020.zip"

#------------------------------------------------------------------------------
# Define the file names
output_zip = external_data_path + "SA2_boundaries/SA2.zip"

#------------------------------------------------------------------------------
# Download the data
urlretrieve(SA2_URL_ZIP, output_zip)

#------------------------------------------------------------------------------
# Extracting the zip file of the geospatial data
# Specifythe zip file name
file_name = external_data_path + "SA2_boundaries/SA2.zip"
  
#------------------------------------------------------------------------------
# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:
    # extracting all the files
    zip.extractall(path = external_data_path + "SA2_boundaries/")

#==============================================================================
# SA2 TOTAL POPULATION DATASET DOWNLOAD
#==============================================================================
# Specify the url
SA2_URL_POP = "https://www.abs.gov.au/statistics/people/population/regional-population/2021/32180DS0001_2001-21.xlsx"

#------------------------------------------------------------------------------
# Define the file names
output = external_data_path + "SA2_total_population/SA2_pop.xlsx"

#------------------------------------------------------------------------------
# Download the data
urlretrieve(SA2_URL_POP, output)

#==============================================================================
# SA2 INCOME DATA DOWNLOAD
#==============================================================================
# Specify the url
SA2_URL_INCOME = "https://www.abs.gov.au/statistics/labour/earnings-and-working-conditions/personal-income-australia/2014-15-2018-19/6524055002_DO001.xlsx"

#------------------------------------------------------------------------------
# Define the file names
output = external_data_path + "SA2_income/SA2_income.xlsx"

#------------------------------------------------------------------------------
# Download the data
urlretrieve(SA2_URL_INCOME, output)

#==============================================================================
# SA2 CENSUS DATA DOWNLOAD
#==============================================================================
# Specify the url
SA2_CENSUS_URL = "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA2_for_AUS_short-header.zip"

#------------------------------------------------------------------------------
# Define the file name
output_csv = external_data_path + "SA2_census/census.zip"

#------------------------------------------------------------------------------
# Download the data
urlretrieve(SA2_CENSUS_URL, output_csv) 

#------------------------------------------------------------------------------
# Opening the zip file in read mode
with ZipFile(output_csv, 'r') as zip:
    # extracting all the files
    zip.extractall(path = external_data_path + "SA2_census/")
#=============================================================================-
# SA2 TO POSTCODE DATA DOWNLOAD
#=============================================================================-
# Specify the url
SA2_POSTCODE_URL = "https://raw.githubusercontent.com/matthewproctor/australianpostcodes/master/australian_postcodes.csv"

#------------------------------------------------------------------------------
# Define the file names
output_csv = external_data_path + "postcode.csv"

#------------------------------------------------------------------------------
# Download the data
urlretrieve(SA2_POSTCODE_URL, output_csv) 

# #============================================================================
# # SA2 TO POSTCODE (for visualisations) DATA DOWNLOAD
# #============================================================================
# # Specify the url
SA2_POSTCODE_URL = "https://raw.githubusercontent.com/schappim/australian-postcodes/master/australian-postcodes.csv"

# #----------------------------------------------------------------------------
# # Define the file names
output_csv = external_data_path + "visualisations_postcodes.csv"

# #----------------------------------------------------------------------------
# # Download the data
urlretrieve(SA2_POSTCODE_URL, output_csv) 

#==============================================================================
# TRANSFORM
#==============================================================================
# PREPROCESSING MERCHANTS DATA
#==============================================================================
# Remove outer brackets in tags
df = tbl_merchants.withColumn("tags", F.regexp_replace("tags", "[\])][\])]", 
        "")) \
        .withColumn("tags", F.regexp_replace("tags", "[\[(][\[(]", "")) 

# separate tags into categories, take rate, and revenue level
# convert take rate to double
tbl_merchants = df.withColumn('categories', F.split(df['tags'], 
        '[)\]], [\[(]').getItem(0)) \
        .withColumn('take_rate', F.split(df['tags'], 
        '[)\]], [\[(]take rate: ').getItem(1).cast("double")) \
        .withColumn('revenue_levels', F.split(df['tags'], 
        '[)\]], [\[(]').getItem(1)) \
        .drop(F.col("tags")) \
        .withColumnRenamed("name", "merchant_name")

# tbl_merchants = tbl_merchants.filter("merchant_abn IS NOT NULL")

#==============================================================================
# PREPROCESSING CONSUMER DATA
#==============================================================================
# Change consumer_id from string to long type
tbl_consumer = tbl_consumer.withColumn("int_consumer_id", 
        tbl_consumer["consumer_id"].cast(LongType())) \
        .drop(F.col("consumer_id"))

#==============================================================================
# PREPROCESSING TRANSACTIONS DATA
#==============================================================================
transactions = transactions.withColumnRenamed("merchant_abn", 
        "trans_merchant_abn") \
        .withColumnRenamed("user_id", "trans_user_id")

#==============================================================================
# PREPROCESSING THE SA2 TOTAL POPULATION DATASET
#==============================================================================
# Read the SA2 total population by districts dataset
population = pd.read_excel(
    external_data_path + "SA2_total_population/SA2_pop.xlsx",
    sheet_name="Table 1")

# Select the relevant rows to filter out uneccessary data
population = population.iloc[8:,:31]

# Define the new column names for better readability
cols = ['state_code', 'state_name', 'GCCSA_code', 'GCCSA_name', 'SA4_code', 
'SA4_name', 'SA3_code', 'SA3_name', 'SA2_code','SA2_name', '2001', '2002', 
'2003','2004','2005','2006', '2007','2008','2009','2010','2011','2012','2013',
'2014','2015','2016','2017','2018','2019', 'population_2020','population_2021']	

# Set the new column names to the dataframe
population.columns = cols

# Select the required columns
population = population[['SA2_code', 'SA2_name','state_code', 'state_name',
'population_2020','population_2021']]

# Filter out the unwanted rows
population.dropna(subset=['state_code'], inplace=True)
population.drop([2466, 2468], inplace=True)

# Checking for null values
population.dropna(inplace=True)

# Save the final curated dataset as csv file
population.to_csv(curated_data_path + "SA2_total_population.csv")

#==============================================================================
# PREPROCESSING SA2 DISTRICT BOUNDARIES DATASET
#==============================================================================
# Loading the data
boundaries = gpd.read_file(
    external_data_path + "SA2_boundaries/SA2_2021_AUST_GDA2020.shp")

# Checking for null values
boundaries.dropna(inplace=True)

# Converting the geometrical objects to latitude and longitude for geospatial 
# visualizations
boundaries['geometry'] = boundaries['geometry'].to_crs(
    "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

# Selecting the required columns
boundaries = boundaries[['SA2_CODE21', 'SA2_NAME21', 'STE_CODE21', 
'STE_NAME21', 'geometry']]

# Renaming the columns
new_columns = ['SA2_code', 'SA2_name', 'state_code', 'state_name', 'geometry']
boundaries.columns = new_columns
# Saving the cleaned dataset
boundaries.to_file(curated_data_path + "boundaries.shp", 
driver='ESRI Shapefile')
# Saving the cleaned dataset
boundaries.to_csv(curated_data_path + "SA2_district_boundaries.csv")

#==============================================================================
# PREPROCESSING SA2 INCOME DATASET
#==============================================================================
# Loading the data
income = pd.read_excel(external_data_path + "SA2_income/SA2_income.xlsx", 
sheet_name="Table 1.4")

# Select the necessary columns
income = income.iloc[6:,[0,1,12,13,14,15,16]]

# Define the new column names for better readability
cols_income = ['SA2_code', "SA2_name", "2014-2015", "2015-2016", "2016-2017", 
"2017-2018",
 "income_2018-2019"]

# Set the new column names to the dataframe
income.columns = cols_income

# Selecting the required columns
income = income[['SA2_code', "SA2_name", "income_2018-2019"]]

# Remove the unwanted values i.e. values that are of type string in the 
# numeric columns 
for index, rows in income.iteritems():
    if index != 'SA2_name':
        for value in rows.values:
            if type(value) == str: 
                income = income[income[index] != value]
                
# Checking for null values
income.dropna(inplace=True)

# Saving the cleaned dataset
income.to_csv(curated_data_path + "SA2_income.csv")
        
#==============================================================================
# PREPROCESSING SA2 CENSUS DATASET
#==============================================================================
# Read the csv file
census = pd.read_csv(
    external_data_path + "SA2_census/2021 Census GCP Statistical Area 2 for AUS/2021Census_G01_AUST_SA2.csv")

# Drop the null values
census.dropna()

# Selecting the required columns
census = census[['SA2_CODE_2021', 'Tot_P_M', 'Tot_P_F', 'Tot_P_P']]

# Renaming the columns
new_cols = ['SA2_code', 'total_males', 'total_females', 'total_persons']
census.columns = new_cols

# Save as a csv file
census.to_csv(curated_data_path + "SA2_census.csv")

#==============================================================================
# PREPROCESSING COVID-19 dataset
#==============================================================================
# Read the data
covid = pd.read_csv(external_data_path + "covid.csv")

# Selecting the required columns
cols = ['date', 'state', 'confirmed']
covid = covid[cols]

# Renaming the columns
new_cols = ['date', 'state_name', 'covid_cases']
covid.columns = new_cols

# Extract the year, month and date from the timestamp
covid['yyyy'] = pd.to_datetime(covid['date']).dt.year
covid['mm'] = pd.to_datetime(covid['date']).dt.month
covid['dd'] = pd.to_datetime(covid['date']).dt.day

# Save the cleaned data to the curated folder
covid.to_csv(curated_data_path + "covid.csv")

#==============================================================================
# PREPROCESSING POSTCODE dataset
#==============================================================================
# Read the data
postcodes = spark.read.option("header", True).csv(
    external_data_path + "postcode.csv")

# Selecting the required columns and renaming 
postcodes = postcodes.select(postcodes.locality.alias("suburb"),
                             postcodes.postcode.alias("postcodes"), 
                             postcodes.SA2_MAINCODE_2016.alias("sa2"),
                             postcodes.long, postcodes.lat)
postcode_nonull = postcodes.na.drop()
postcode_nonull.count()

# Drop duplicate postcodes in postcode SA2 mapping dataset
distinct_postcodes = postcode_nonull.dropDuplicates(["postcodes"])

# Save the cleaned data to the curated folder
distinct_postcodes.write.mode("overwrite").csv(
    curated_data_path + "postcode.csv")

#==============================================================================
# LOAD
#==============================================================================
# PERFORMING INTERNAL DATA JOINS
#==============================================================================
# Join transactions to user details
trans_user = transactions.join(user_details,
transactions.trans_user_id ==  user_details.user_id,"inner")

# Join consumer to above data
add_consumer = tbl_consumer.join(trans_user, 
tbl_consumer.int_consumer_id ==  trans_user.consumer_id,"inner")

# Join merchants to above data
final_join = tbl_merchants.join(add_consumer, 
        tbl_merchants.merchant_abn == add_consumer.trans_merchant_abn, 
        "full_outer") \
        .drop(F.col("int_consumer_id")) \
        .drop(F.col("trans_user_id"))

#==============================================================================
# JOIN POSTCODE TO SA2 MAPPING
#==============================================================================

final_join2 = final_join.join(distinct_postcodes, 
        final_join.postcode == distinct_postcodes.postcodes, "inner") \
        .drop(F.col("postcode"))

final_join2 = final_join2.withColumn("int_sa2", 
    final_join2["sa2"].cast(IntegerType())).drop(F.col("sa2"))

#==============================================================================
# JOIN ABS DATA TO INTERNAL DATA
#==============================================================================
# Retrive the required data
population = pd.read_csv(curated_data_path + "SA2_total_population.csv")
income = pd.read_csv(curated_data_path + "SA2_income.csv")
census = pd.read_csv(curated_data_path + "SA2_census.csv")

# ----------------------------------------------------------------------------
# Remove the first unwanted column from ABS datasets
population = population.iloc[:,1:]
income = income.iloc[:,1:]
census = census.iloc[:,1:]

# ----------------------------------------------------------------------------
# Perform inner join on income and census dataset
income_census = pd.merge(income, census, on = 'SA2_code')

# ----------------------------------------------------------------------------
# Make a final dataset from the the merged datasets
SA2_datasets = pd.merge(income_census, population, on = ['SA2_code', 
'SA2_name'])

# ----------------------------------------------------------------------------
# Convert SA2_datasets to pyspark dataframe
SA2_datasets_spark = spark.createDataFrame(SA2_datasets)

#==============================================================================
# CREATE A FINAL DATASET
#==============================================================================
final_join3 = final_join2.join(SA2_datasets_spark, 
        final_join2.int_sa2 == SA2_datasets_spark.SA2_code, "inner") \
        .drop(F.col("postcode"))


#==============================================================================
# ADD FRAUD DATA TO A FINAL DATASET
#==============================================================================

# adding consumer fraud data
final_join3 = final_join3.alias("a").join(fraud_consumer.alias("b"), 
            (final_join3.user_id == fraud_consumer.user_id) \
            & (final_join3.order_datetime == fraud_consumer.order_datetime), 
            "left_outer") \
            .select("a.*","b.fraud_probability")

# rename fraud probability for consumer fraud data
final_join3 = final_join3.withColumnRenamed('fraud_probability', 
'fraud_probability_consumer')

# adding merchant fraud data
final_join3 = final_join3.alias("a").join(fraud_merchants.alias("b"), 
        (final_join3.merchant_abn == fraud_merchants.merchant_abn) \
        & (final_join3.order_datetime == fraud_merchants.order_datetime), 
        "left_outer") \
        .select("a.*","b.fraud_probability")
    
# rename fraud probability for merchant fraud data
final_join3 = final_join3.withColumnRenamed('fraud_probability', 
'fraud_probability_merchant')

# apply double type to fraud probability
from pyspark.sql.types import DoubleType
final_join3 = final_join3.withColumn("fraud_probability_consumer",
                    col("fraud_probability_consumer").cast(DoubleType())) \
                    .withColumn("fraud_probability_merchant",
                    col("fraud_probability_merchant").cast(DoubleType()))

# Add small values to the null value
SMALL_PROB = 0.01

final_join3 = final_join3.fillna(SMALL_PROB, 
subset=["fraud_probability_merchant", "fraud_probability_consumer"])


