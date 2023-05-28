
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
import fraud_feature
import geopandas as gpd

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
# Plot the total male and female transactions
#print("The distribution of transactions according to gender")
fig1, ax1 = plt.subplots(figsize=(12,7))
genders = outlier.internal4.select("gender")
genderspd = genders.toPandas()
sns.countplot(data=genderspd, x="gender")
ax1.ticklabel_format(style='plain', axis='y')
ax1.set_xlabel("Gender")
ax1.set_title("Number of transactions per gender")
plt.savefig(visualisation_path + "Gender transactions.jpg",dpi=300, 
bbox_inches='tight')

#------------------------------------------------------------------------------
# Distribution of total revenue for each merchant from online purchases
#print("The distribution of total revenue across all the merchants")
selected_columns = outlier.internal4.select("merchant_abn","dollar_value")
aggregated_revenue = selected_columns.groupby("merchant_abn").sum(
    "dollar_value")
aggregated_revenue_pd = aggregated_revenue.toPandas()
total_revenue = aggregated_revenue_pd['sum(dollar_value)']

fig2, ax2 = plt.subplots(figsize=(12,7))
sns.boxplot(total_revenue)
ax2.set_xlabel("Total revenue per merchant")
ax2.set_title("Distribution of merchant revenue")
plt.savefig(visualisation_path + "Revenue distribution.jpg",dpi=300, 
bbox_inches='tight')

#------------------------------------------------------------------------------
# Distributions of transactions by state
#print("Number of transactions per Australian state")
state = outlier.internal4.select("state")
statepd = state.toPandas()

fig3, ax3 = plt.subplots(figsize=(12,7))
sns.countplot(data=statepd, x="state")
ax3.ticklabel_format(style='plain', axis='y')
ax3.set_xlabel("State")
ax3.set_title("Number of transactions per Victorian state")
plt.savefig(visualisation_path + "Transactions per state.jpg",dpi=300, 
bbox_inches='tight')

#------------------------------------------------------------------------------
# Number of transactions made per month

# Read the tagged model
tagged_merchants_sdf = spark.read.parquet(
    curated_data_path + "tagged_merchants.parquet")

# -----------------------------------------------------------------------------
# Join the final dataset to the tagged model
tagged_merchants_sdf = tagged_merchants_sdf.withColumnRenamed('merchant_abn',

    'tagged_merchant_abn'
)
# -----------------------------------------------------------------------------
# Calculate the BNPL earnings 
outlier.internal4.createOrReplaceTempView("join")
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


transactions_new_sdf = joint.withColumn("Year", 
date_format('order_datetime', 'yyyy'))

transactions_new_sdf = transactions_new_sdf.withColumn("Month", 
date_format('order_datetime', 'MMMM'))


transactions_new_sdf.createOrReplaceTempView("temp_view")

trans_2021 = spark.sql(""" 

SELECT Month, category, COUNT(merchant_abn) as transactions_2021
FROM temp_view
WHERE Year == '2021'
GROUP BY Month, category

""")

trans_2022 = spark.sql(""" 

SELECT Month, category, COUNT(merchant_abn) as transactions_2022
FROM temp_view
WHERE Year == '2022'
GROUP BY Month, category

""")


trans_2021_df = trans_2021.toPandas()
trans_2022_df = trans_2022.toPandas()


month_2021 = ['February', 'March', 'April', 'May', 'June', 'July', 'August', 
'September', 'October', 'November','December']
month_2022 = ['Janaury', 'February', 'March', 'April', 'May', 'June', 'July', 
'August', 'September']



# Order the month column 


trans_2021_df['Month'] = pd.Categorical(trans_2021_df['Month'], 
categories = month_2021, ordered=True)
trans_2021_df.sort_values(by = "Month", inplace = True)

trans_2022_df['Month'] = pd.Categorical(trans_2022_df['Month'], 
categories = month_2022, ordered=True)
trans_2022_df.sort_values(by = "Month", inplace = True)



fig4, ax4 = plt.subplots(figsize=(12,7))
sns.lineplot(data=trans_2021_df, x="Month", y="transactions_2021", 
hue="category")
ax4.set_ylabel("Number of transactions in 2021")
ax4.set_title("Number of transactions per category in 2021")
plt.savefig(visualisation_path + "Transactions per category 2021.jpg",dpi=300, 
bbox_inches='tight')


fig5, ax5 = plt.subplots(figsize=(12,7))
sns.lineplot(data=trans_2022_df, x="Month", y="transactions_2022", 
hue="category")
ax5.set_ylabel("Number of transactions in 2022")
ax5.set_title("Number of transactions per category in 2022")
plt.savefig(visualisation_path + "Transactions per category 2022.jpg",dpi=300, 
bbox_inches='tight')
#------------------------------------------------------------------------------
# Revenue per category in 2021 and 2022
transactions_new_sdf.createOrReplaceTempView("temp_view")

revenue_2021 = spark.sql(""" 

SELECT Month, category, 
    SUM(dollar_value-((take_rate/100)*dollar_value)) as revenue_2021
FROM temp_view
WHERE Year == '2021'
GROUP BY Month, category

""")

revenue_2022 = spark.sql(""" 

SELECT Month, category, 
    SUM(dollar_value-((take_rate/100)*dollar_value)) as revenue_2022
FROM temp_view
WHERE Year == '2022'
GROUP BY Month, category

""")


revenue_2021_df = revenue_2021.toPandas()
revenue_2022_df = revenue_2022.toPandas()


month_2021 = ['February', 'March', 'April', 'May', 'June', 'July', 'August', 
'September', 'October', 'November','December']
month_2022 = ['Janaury', 'February', 'March', 'April', 'May', 'June', 'July', 
'August', 'September']



# Order the month column 


revenue_2021_df['Month'] = pd.Categorical(revenue_2021_df['Month'], 
categories = month_2021, ordered=True)
revenue_2021_df.sort_values(by = "Month", inplace = True)

revenue_2022_df['Month'] = pd.Categorical(revenue_2022_df['Month'], 
categories = month_2022, ordered=True)
revenue_2022_df.sort_values(by = "Month", inplace = True)



fig6, ax6 = plt.subplots(figsize=(12,7))
sns.lineplot(data=revenue_2021_df, x="Month", y="revenue_2021", 
hue="category")
ax6.set_ylabel("Amount of revenue in 2021")
ax6.ticklabel_format(style='plain', axis='y')
ax6.set_title("Amount of revenue per category in 2021")
plt.savefig(visualisation_path + "Revenue per category 2021.jpg",dpi=300, 
bbox_inches='tight')

fig7, ax7 = plt.subplots(figsize=(12,7))
sns.lineplot(data=revenue_2022_df, x="Month", y="revenue_2022", 
hue="category")
ax7.set_ylabel("Amount of revenue in 2022")
ax7.ticklabel_format(style='plain', axis='y')
ax7.set_title("Amount of revenue per category in 2022")
plt.savefig(visualisation_path + "Revenue per category 2022.jpg",dpi=300, 
bbox_inches='tight')

#------------------------------------------------------------------------------
# Calculate the total BNPL earnings
transactions_new_sdf.createOrReplaceTempView("temp_view")

BNPL_2021 = spark.sql(""" 

SELECT Month, category, SUM((take_rate/100)*dollar_value) as BNPL_2021
FROM temp_view
WHERE Year == '2021'
GROUP BY Month, category

""")

BNPL_2022 = spark.sql(""" 

SELECT Month, category, SUM((take_rate/100)*dollar_value) as BNPL_2022
FROM temp_view
WHERE Year == '2022'
GROUP BY Month, category

""")


BNPL_2021_df = BNPL_2021.toPandas()
BNPL_2022_df = BNPL_2022.toPandas()


month_2021 = ['February', 'March', 'April', 'May', 'June', 'July', 'August', 
'September', 'October', 'November','December']
month_2022 = ['Janaury', 'February', 'March', 'April', 'May', 'June', 'July', 
'August', 'September']



# Order the month column 


BNPL_2021_df['Month'] = pd.Categorical(BNPL_2021_df['Month'], 
categories = month_2021, ordered=True)
BNPL_2021_df.sort_values(by = "Month", inplace = True)

BNPL_2022_df['Month'] = pd.Categorical(BNPL_2022_df['Month'], 
categories = month_2022, ordered=True)
BNPL_2022_df.sort_values(by = "Month", inplace = True)



fig8, ax8 = plt.subplots(figsize=(12,7))
sns.lineplot(data=BNPL_2021_df, x="Month", y="BNPL_2021", 
hue="category")
ax8.set_ylabel("BNPL earnings in $ in 2021")
ax8.set_title("Amount of BNPL earnings in 2021")
plt.savefig(visualisation_path + "Transactions per category 2021.jpg",dpi=300, 
bbox_inches='tight')


fig9, ax9 = plt.subplots(figsize=(12,7))
sns.lineplot(data=BNPL_2022_df, x="Month", y="BNPL_2022", 
hue="category")
ax9.set_ylabel("BNPL earnings in $ in 2022")
ax9.set_title("Amount of BNPL earnings in 2022")
plt.savefig(visualisation_path + "Transactions per category 2022.jpg",dpi=300, 
bbox_inches='tight')


#------------------------------------------------------------------------------
# Geospatial visualizations of number of transactions by post code
num_transactions_by_postcode = ETL.final_join3.groupBy(['postcodes', 'suburb', 
'long', 'lat']).count()
# conver transactions location to pandas df
num_transactions_by_postcode_pdf = num_transactions_by_postcode.toPandas()

aus_coords = [-25.2744, 133.7751]
m = folium.Map(aus_coords, tiles='OpenStreetMap', zoom_start=4.5)

for index, row in num_transactions_by_postcode_pdf.iterrows():
    if row['count'] >= 10000:
        marker_color = 'darkred'
        fill_color = 'darkred'
    elif row['count'] >= 5000:
        marker_color = 'red'
        fill_color = 'red'
    elif row['count'] >= 500:
        marker_color = 'darkorange'
        fill_color = 'darkorange'
    elif row['count'] >= 100:
        marker_color = 'orange'
        fill_color = 'orange'
    elif row['count'] <= 50 :
        marker_color = 'yellow'
        fill_color = 'yellow'
    else:
        marker_color='darkpurple'
        fill_color = 'darkpurple'
        
    folium.Circle(
          location=[row['lat'], row['long']],
          popup= 'Number of transactions: ' +str(row['count']),
          tooltip=row['suburb'],
          radius=row['count'],
          color=marker_color,
          fill=True,
          fill_color=fill_color,
       ).add_to(m)
m.save(visualisation_path + "bubble_plot_num_transactions_by_location.html")

#------------------------------------------------------------------------------
## The following was modified from MAST30034 Tutorial 2
# read in shape file
sf = gpd.read_file("../data/curated/boundaries.shp")
gdf = gpd.GeoDataFrame(sf)
gdf['SA2_code'] = gdf['SA2_code'].astype(int)
geoJSON = gdf[['SA2_code', 'geometry']].drop_duplicates('SA2_code').to_json()
### Locations of fraudulent consumers
# aggregate data by postcode and consumer fraud probability
consumer_fraud_postcodes = fraud_feature.model_with_fraud.select('postcodes', 
'SA2_code', 'fraud_probability_consumer')
consumer_fraud_postcodes = consumer_fraud_postcodes.groupBy('postcodes', 
'SA2_code')\
.agg(F.sum('fraud_probability_consumer').alias('sum_consumer_fraud'))
consumer_fraud_pdf = consumer_fraud_postcodes.toPandas()
# join shape file to consumer fraud data by sa2 code
consumer_df = consumer_fraud_pdf \
    .merge(gdf[['SA2_code', 'geometry']], left_on='SA2_code', 
    right_on='SA2_code')
consumer_df.head(5)
aus_coords = [-25.2744, 133.7751]
consumer_fraud_map = folium.Map(aus_coords, tiles="Stamen Terrain", 
zoom_start=4.5)

# add average tips
c = folium.Choropleth(
    geo_data=geoJSON, # geoJSON 
    name='choropleth', # name of plot
    data=consumer_df, # data source
    columns=['SA2_code','sum_consumer_fraud'], # the columns required
    key_on='properties.SA2_code', # this is from the geoJSON's properties
    fill_color='YlOrRd', # color scheme
    nan_fill_color='black',
    legend_name='Total consumer fraud probability by SA2 region'
)
c.add_to(consumer_fraud_map)
consumer_fraud_map.save(visualisation_path + "consumer_fraud_map.html")
### Locations of fraudulent merchants
# aggregate data by postcode and merchant fraud probability
merchant_fraud_postcodes = fraud_feature.model_with_fraud.select('postcodes', 
'SA2_code', 'fraud_probability_merchant')
merchant_fraud_postcodes = merchant_fraud_postcodes.groupBy('postcodes', 
'SA2_code')\
.agg(F.sum('fraud_probability_merchant').alias('sum_merchant_fraud'))

merchant_fraud_pdf = merchant_fraud_postcodes.toPandas()
# join shape file to consumer fraud data by sa2 code
merchant_df = merchant_fraud_pdf \
    .merge(gdf[['SA2_code', 'geometry']], left_on='SA2_code', 
    right_on='SA2_code')

aus_coords = [-25.2744, 133.7751]
merchant_fraud_map = folium.Map(aus_coords, tiles="Stamen Terrain", 
zoom_start=4.5)

# add average tips
d = folium.Choropleth(
    geo_data=geoJSON, # geoJSON 
    name='choropleth', # name of plot
    data=merchant_df, # data source
    columns=['SA2_code','sum_merchant_fraud'], # the columns required
    key_on='properties.SA2_code', # this is from the geoJSON's properties
    fill_color='YlOrRd', # color scheme
    nan_fill_color='black',
    legend_name='Total merchant fraud probability by SA2 region'
)
d.add_to(merchant_fraud_map)
merchant_fraud_map.save(visualisation_path + "merchant_fraud_map.html")








