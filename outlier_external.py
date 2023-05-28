#==============================================================================
import matplotlib.pyplot as plt
import sys, json
import outlier
import pandas as pd
import seaborn as sns
import folium
from pyspark.sql.functions import date_format
from pyspark.sql import SparkSession, functions as F
import ETL
import outlier
import numpy as np

#==============================================================================
# Create a spark session
spark = (
    SparkSession.builder.appName("MAST30034 Project 2 part 4")
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

visualisation_path = PATHS['visualisation_path']
#------------------------------------------------------------------------------

# summary statistics
print("Summary statistics for the Census data - ")
print("\n")
print(ETL.census.describe())
print("\n")


print("Summary statistics for the Income data - ")
print("\n")
print(ETL.income.head())
print("\n")

# summary statistics income_2018-2019
print("Summary statistics for the Income data for 2018-2019 - ")
print("\n")
print(ETL.income['income_2018-2019'].describe())
print("\n")

# Check which SA2 regions have negative sum of income for that region
print("Check which SA2 regions have negative sum of income for that region - ")
print("\n")
print(ETL.income.loc[ETL.income['income_2018-2019'] < 0])
# Check if this SA2 region is in the final join
print("\n")
print("Check if this SA2 region is in the final dataset")
print(outlier.internal4.filter(outlier.internal4.SA2_code == 114011275
).collect())
print("\n")


print("Summary statistics for the Population data - ")
print("\n")
print(ETL.population.head())
print("\n")

# descriptive stats for 2020 in each SA2 region
print("Summary statistics for the population in 2020 per SA2 code data - ")
print("\n")
print(ETL.population['population_2020'].describe())
print("\n")

print("Summary statistics for the population in 2021 per SA2 code data - ")
print("\n")
# descriptive stats for 2021 in each SA2 region
print(ETL.population['population_2021'].describe())
print("\n")

print("Distribution of Income")
fig1, ax1 = plt.subplots()
# income distribution
ETL.income['income_2018-2019'].astype(np.double).plot()
ax1.set_xlabel("Income distribution for 2018-2019")
ax1.ticklabel_format(style='plain', axis='y')
plt.savefig(visualisation_path + "Income distribution.jpg",dpi=300, 
bbox_inches='tight')