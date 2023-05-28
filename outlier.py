#==============================================================================
import sys, json
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
import pandas as pd
import ETL

#------------------------------------------------------------------------------
# Create a spark session
spark = (
    SparkSession.builder.appName("MAST30034 Project 2 part 3")
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

#==============================================================================
### REMOVE OUTLIERS as detected in outlier analysis notebook
#------------------------------------------------------------------------------
# Excluding transactions with no merchants

internal1 = ETL.final_join3.filter("merchant_abn IS NOT NULL")

# Excluding transactions with $0
internal2 = internal1.filter(internal1.dollar_value > 0)

# Excluding merchants with no transactions and record merchant name
merchants_no_trans = internal2.filter("consumer_id IS NULL")
internal3 = internal2.filter("consumer_id IS NOT NULL")
internal4 = internal3.filter("gender IS NOT NULL")

data = {
    'Outlier removal': ['Original count', 'Valid Merchant ABN', 
                        'Transactions with non $0', 'Valid customer ID',
                        'Non null values for gender'],
    'Count after outlier removal': [ETL.final_join3.count(), internal1.count(),
                                    internal2.count(), internal3.count(),
                                    internal4.count()]
}

df = pd.DataFrame(data)



internal4.write.mode("overwrite").parquet(
    curated_data_path + "full_join.parquet")