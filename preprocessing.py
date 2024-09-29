import pandas as pd
import matplotlib.pyplot as plt
import sweetviz
from AutoClean import AutoClean
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from clusteval import clusteval
import numpy as np

kit =  pd.read_csv(r"G:\class is started\PROJECT\given dataset\KIT (1)\KIT.csv")

from sqlalchemy import create_engine

# Credentials to connect to Database
user = 'root'  # user name
pw = 'patilasp'  # password
db = 'proj'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


# to_sql() - function to push the dataframe onto a SQL table.
kit.to_sql('kititem', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from kititem;'
df = pd.read_sql_query(sql, engine)




# Selecting required columns
required_columns = ['4/1/2021 0:00','5/1/2021 0:00','6/1/2021 0:00','7/1/2021 0:00','8/1/2021 0:00','9/1/2021 0:00','10/1/2021 0:00','11/1/2021 0:00','12/1/2021 0:00','1/1/2022 0:00','2/1/2022 0:00','3/1/2022 0:00','4/1/2022 0:00','5/1/2022 0:00','6/1/2022 0:00','7/1/2022 0:00','8/1/2022 0:00','9/1/2022 0:00','10/1/2022 0:00','11/1/2022 0:00','12/1/2022 0:00','1/1/2023 0:00','2/1/2023 0:00','3/1/2023 0:00','4/1/2023 0:00','5/1/2023 0:00','6/1/2023 0:00','7/1/2023 0:00','8/1/2023 0:00','9/1/2023 0:00','10/1/2023 0:00','11/1/2023 0:00','12/1/2023 0:00'
]
new_df = df[required_columns]

# Display the new DataFrame

print(new_df)




# Assuming new_df is your DataFrame with no index
transposed_df = new_df.transpose().reset_index()

# Display the transposed DataFrame
print(transposed_df)


#RENAMING

df_1 = transposed_df.rename(columns={'index': 'date'})

print(df_1)

df1 = df_1.set_index('date')

# Selecting required columns
required_columns2 = ['KIT ITEM']
new_df1 = df[required_columns2]
# Display the new DataFrame
print(new_df1)

# Assuming new_df is your DataFrame with no index
transposed_df1 = new_df1.transpose().reset_index()

transposed_df1.columns
# Display the transposed DataFrame
print(transposed_df1)



final_df = pd.concat([transposed_df1, df1])

# Display the concatenated DataFrame
print(final_df)

#first column is deleted using this

final_df = final_df.rename_axis("Date")

#drops first column which contain nan valuue"KIT item heading"
final_df = final_df.drop(final_df.columns[0], axis=1)


final_df.columns = ['KIT0000896', 'KIT0000897', 'KIT0000613', 'KIT0000634', 'KIT0000638', 'KIT0000642', 'KIT0000645', 'KIT0000646', 'KIT0000647', 'KIT0000725', 'KIT0000735', 'KIT0000737', 'KIT0000738', 'KIT0000741', 'KIT0000792', 'KIT0000813', 'KIT0000815', 'KIT0000285', 'KIT0000286', 'KIT0000949', 'KIT0001052', 'KIT0001053', 'KIT0001054', 'KIT0000937', 'KIT0000938', 'KIT0000939', 'KIT0001016', 'KIT0001038', 'KIT0001039', 'KIT0000894', 'KIT0000895', 'KIT0001157', 'KIT0001158', 'KIT0001134', 'KIT0001135', 'KIT0001136', 'KIT0001137', 'KIT0001151', 'KIT0001152', 'KIT0001153', 'KIT0000219', 'KIT0001061', 'KIT0000933', 'KIT0000627', 'KIT0001066', 'KIT0000580', 'KIT0000581', 'KIT0000862', 'KIT0000928', 'KIT0000365', 'KIT0000366', 'KIT0001042', 'KIT0000370', 'KIT0000445', 'KIT0000609', 'KIT0000905', 'KIT0000968', 'KIT0000468', 'KIT0000524', 'KIT0000563', 'KIT0000564', 'KIT0000565', 'KIT0000566', 'KIT0000569', 'KIT0000877', 'KIT0000994', 'KIT0000995', 'KIT0000996', 'KIT0001012', 'KIT0001158', 'KIT0000975', 'KIT0000976', 'KIT0000977', 'KIT0000978', 'KIT0001107', 'KIT0001108', 'KIT0001134', 'KIT0001135', 'KIT0001136', 'KIT0001137', 'KIT0001151', 'KIT0001152', 'KIT0001153', 'KIT0000199', 'KIT0000783', 'KIT0001018', 'KIT0001019', 'KIT0001020', 'KIT0001021', 'KIT0001022', 'KIT0001029', 'KIT0001031', 'KIT0001065', 'KIT0001112', 'KIT0001121', 'KIT0001139', 'KIT0001145', 'KIT0001146', 'KIT0001147', 'KIT0001046', 'KIT0000024', 'KIT0000029', 'KIT0000030', 'KIT0000032', 'KIT0000417', 'KIT0000997', 'KIT0000143', 'KIT0000144', 'KIT0000145', 'KIT0000146', 'KIT0000660', 'KIT0001140', 'KIT0000868', 'KIT0001109', 'KIT0000519', 'KIT0001102', 'KIT0001030', 'KIT0000803', 'KIT0000746', 'KIT0000748', 'KIT0000749', 'KIT0000750', 'KIT0000805', 'KIT0000864', 'KIT0000865', 'KIT0000866', 'KIT0000867', 'KIT0000869', 'KIT0000870', 'KIT0000871', 'KIT0001063', 'KIT0000836', 'KIT0001103', 'KIT0000854', 'KIT0000855', 'KIT0001150', 'KIT0001154', 'KIT0001014', 'KIT0000608', 'KIT0000212', 'KIT0001050', 'KIT0001051', 'KIT0001156', 'KIT0001159', 'KIT0001160', 'KIT0000467', 'KIT0000696', 'KIT0000943', 'KIT0001118', 'KIT0001119', 'KIT0000170', 'KIT0001027', 'KIT0001028', 'KIT0001093', 'KIT0001097', 'KIT0001098', 'KIT0000432', 'KIT0000860', 'KIT0001138', 'KIT0000004', 'KIT0000008', 'KIT0000010', 'KIT0000012', 'KIT0000013', 'KIT0000194', 'KIT0000195', 'KIT0000712', 'KIT0000948', 'KIT0000953', 'KIT0000958', 'KIT0000984', 'KIT0000985', 'KIT0001013', 'KIT0001015', 'KIT0001058', 'KIT0001070', 'KIT0001099', 'KIT0001100', 'KIT0001111', 'KIT0001116', 'KIT0001142', 'KIT0001143', 'KIT0000582', 'KIT0001024', 'KIT0001117', 'KIT0001083', 'KIT0001084', 'KIT0001085', 'KIT0001086', 'KIT0000577', 'KIT0001017', 'KIT0000435', 'KIT0000436', 'KIT0000437', 'KIT0000438', 'KIT0000439', 'KIT0000649', 'KIT0001162', 'KIT0000558', 'KIT0000296', 'KIT0000297', 'KIT0000355', 'KIT0000674', 'KIT0001069', 'KIT0000297', 'KIT0000471', 'KIT0000503', 'KIT0000674', 'KIT0000203', 'KIT0000204', 'KIT0000460', 'KIT0000461', 'KIT0000966', 'KIT0001144', 'KIT0001155', 'KIT0000959', 'KIT0000560', 'KIT0001059', 'KIT0001032', 'KIT0001033', 'KIT0001068', 'KIT0000215', 'KIT0000892', 'KIT0001067', 'KIT0001101', 'KIT0001096', 'KIT0001095', 'KIT0001034', 'KIT0001035', 'KIT0000990', 'KIT0000991', 'KIT0000945', 'KIT0000986', 'KIT0001034', 'KIT0001035', 'KIT0000526', 'KIT0001114', 'KIT0000294', 'KIT0000414', 'KIT0001037', 'KIT0000181', 'KIT0000181', 'KIT0000084', 'KIT0000174', 'KIT0000175', 'KIT0000227', 'KIT0000228', 'KIT0000287', 'KIT0000353', 'KIT0000587', 'KIT0000777', 'KIT0000784', 'KIT0000936', 'KIT0001044', 'KIT0001073', 'KIT0001074', 'KIT0000016', 'KIT0000123', 'KIT0000147', 'KIT0000177', 'KIT0000256', 'KIT0000298', 'KIT0001122', 'KIT0001123', 'KIT0001124', 'KIT0001125', 'KIT0001126', 'KIT0001127', 'KIT0001128', 'KIT0001129', 'KIT0001130', 'KIT0001131', 'KIT0001132', 'KIT0001161', 'KIT0000942', 'KIT0000929', 'KIT0000930', 'KIT0000935', 'KIT0001001', 'KIT0001002', 'KIT0001003', 'KIT0001004', 'KIT0001006', 'KIT0001007', 'KIT0001008', 'KIT0000162', 'KIT0000201', 'KIT0000207', 'KIT0000209', 'KIT0000534', 'KIT0000884', 'KIT0000962', 'KIT0001094', 'KIT0001006', 'KIT0001007', 'KIT0000268', 'KIT0001062', 'KIT0001000', 'KIT0000878', 'KIT0000879', 'KIT0000880', 'KIT0000885', 'KIT0000886', 'KIT0000887', 'KIT0000888', 'KIT0001025', 'KIT0001026']


final_df = final_df.drop(final_df.index[0], axis=0)

                            

                #### Number of unique KIT ITEMS: 292 ####
                
uniq_item = df ['KIT ITEM'].nunique()

print("Number of unique KIT ITEMS:", uniq_item)



                #Identify duplicate column names

duplicate_columns = final_df.columns[final_df.columns.duplicated()]
print("Duplicate columns:", duplicate_columns)


                #Aggregate duplicate columns

count_final_df = final_df.groupby(level=0, axis=1).count()

                ## Remove duplicate
final_df = final_df.loc[:, ~final_df.columns.duplicated()]


                #filling nan values with 0
final_df = final_df.fillna(0)


# Define threshold 
# Columns contain 80% zeros will be removed
zero_threshold = 0.20 

#Calculate proportion of zeros in each column
zero_proportion = (final_df == 0).mean()

# Filter columns based on zero proportion threshold
clm_with_Zero = zero_proportion[zero_proportion > zero_threshold].index

#Remove columns with high proportion of zeros 
final_df = final_df.drop(columns=clm_with_Zero)

final_df.to_csv('final_DF.csv', index=False)

                    ##### EDA #####

# Calculate mean, median, mode, kurtosis, skewness, min, max, standard deviation, and variance
stats = final_df.agg(['mean', 'median', lambda x: x.mode().iloc[0], 'kurtosis', 'skew', 'min', 'max', 'std', 'var']).transpose()

# Rename the columns for better readability
stats.columns = ['Mean', 'Median', 'Mode', 'Kurtosis', 'Skewness', 'Min', 'Max', 'Standard Deviation', 'Variance']

# Display the statistics
print(stats)


#Saving the EDA Processing file in csv format
stats.to_csv('BI_moment.csv', index=True)

    ### Preprocessing ###
    

# Check for missing values
missing_values = final_df.isna().sum()

# Display the count of missing values for each column
print(missing_values)


##  For treating missing values we use Random imputation  ##

import pandas as pd
import numpy as np

# Assuming final_df is your DataFrame with 0 values
# Replace 0 values with a random sample from the non-zero values in each column
final_df_random_imputed = final_df.apply(lambda col: col.replace(0, np.random.choice(col[col != 0]), inplace=True))

# Check if there are any remaining 0 values after imputation
zero_values = (final_df_random_imputed == 0).sum()
print("Remaining 0 values after random imputation:")
print(zero_values)


            ##Outlier treatment##
        ### symmetric Winsorization ### 5% of the lowest values and the 5% of the highest values in each column are replaced with values at the 5th and 95th percentiles ##
from scipy.stats import mstats

# Winsorize each column in the DataFrame
final_df_winsorized = final_df.apply(lambda col: mstats.winsorize(col, limits=[0.05, 0.05]))

# Print the winsorized DataFrame
print(final_df_winsorized)



    #### ADF and KPSS test ####

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# Assuming final_df is your DataFrame containing the columns
for column in final_df.columns:
    # Perform ADF test
    result_adf = adfuller(final_df[column])
    adf_statistic = result_adf[0]
    adf_pvalue = result_adf[1]
    adf_critical_values = result_adf[4]

    # Perform KPSS test
    result_kpss = kpss(final_df[column])
    kpss_statistic = result_kpss[0]
    kpss_pvalue = result_kpss[1]
    kpss_critical_values = result_kpss[3]

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(final_df[column], label=f"{column} Time Series")

    # ADF test results
    ax.axhline(y=adf_statistic, color='r', linestyle='--', label=f"ADF Statistic: {adf_statistic:.2f}")
    ax.axhline(y=adf_critical_values['1%'], color='g', linestyle='-', label="ADF 1% Critical Value")
    ax.axhline(y=adf_critical_values['5%'], color='g', linestyle='--', label="ADF 5% Critical Value")
    ax.axhline(y=adf_critical_values['10%'], color='g', linestyle=':', label="ADF 10% Critical Value")

    # KPSS test results
    ax.axhline(y=kpss_statistic, color='b', linestyle='-.', label=f"KPSS Statistic: {kpss_statistic:.2f}")
    ax.axhline(y=kpss_critical_values['1%'], color='m', linestyle='-', label="KPSS 1% Critical Value")
    ax.axhline(y=kpss_critical_values['5%'], color='m', linestyle='--', label="KPSS 5% Critical Value")
    ax.axhline(y=kpss_critical_values['10%'], color='m', linestyle=':', label="KPSS 10% Critical Value")

    ax.set_title(f"ADF and KPSS Tests for {column}")
    ax.legend()
    plt.show()
    
    
            ### segrigating stationary and non stationary columns ###

# Lists to store stationary and non-stationary column names
stationary_columns = []
non_stationary_columns = []

# Assuming final_df is your DataFrame containing the columns
for column in final_df.columns:
    # Perform ADF test
    result_adf = adfuller(final_df[column])
    adf_pvalue = result_adf[1]

    # Perform KPSS test
    result_kpss = kpss(final_df[column])
    kpss_pvalue = result_kpss[1]

    # Categorize columns based on p-values
    if adf_pvalue < 0.05 and kpss_pvalue > 0.05:
        stationary_columns.append(column)
    else:
        non_stationary_columns.append(column)

# Count the number of stationary and non-stationary columns
num_stationary = len(stationary_columns)
num_non_stationary = len(non_stationary_columns)

# Print the counts
print("Number of Stationary Columns:", num_stationary)
print("Number of Non-Stationary Columns:", num_non_stationary)



# Assuming stationary_columns contains the names of stationary columns
stationary_df = final_df[stationary_columns].copy()

stationary_df.to_csv('stationary_df.csv', index=False)

non_stationary_df = final_df[non_stationary_columns].copy()

non_stationary_df.to_csv('non_stationary_df.csv', index=False)



##converting non_stationary data to stationary by using differencing ##

import pandas as pd

# Assuming your DataFrame is named final_df
# Assuming all your columns are in final_df DataFrame

# List of non-stationary columns
non_stationary_cols = ['KIT0000647', 'KIT0000285', 'KIT0000895', 'KIT0000928', 'KIT0000365', 'KIT0000366', 'KIT0001042', 'KIT0000445', 'KIT0000609', 'KIT0000905', 'KIT0000968', 'KIT0000468', 'KIT0000563', 'KIT0000565', 'KIT0000994', 'KIT0000995', 'KIT0000978', 'KIT0000417', 'KIT0000143', 'KIT0000144', 'KIT0000146', 'KIT0000660', 'KIT0000868', 'KIT0000519', 'KIT0001030', 'KIT0001014', 'KIT0000467', 'KIT0000170', 'KIT0001027', 'KIT0001028', 'KIT0000432', 'KIT0000860', 'KIT0000194', 'KIT0000948', 'KIT0000953', 'KIT0001013', 'KIT0000582', 'KIT0001024', 'KIT0000296', 'KIT0000297', 'KIT0000355', 'KIT0000503', 'KIT0000203', 'KIT0000204', 'KIT0000560', 'KIT0000990', 'KIT0000294', 'KIT0001037', 'KIT0000353', 'KIT0000123', 'KIT0000298', 'KIT0000942', 'KIT0000201', 'KIT0000885', 'KIT0000886', 'KIT0000887', 'KIT0000888', 'KIT0001025']

# Convert non-stationary columns to stationary
for col in non_stationary_cols:
    final_df[col] = final_df[col].diff().fillna(0)

print(final_df)

final_df.to_csv('all_stationary.csv', index=False)




from pmdarima import auto_arima

# Filter columns that start with "KIT"
kit_columns = [col for col in final_df.columns if col.startswith('KIT')]

# Fit AutoARIMA model for each column
for column in kit_columns:
    if column in final_df.columns:
        model = auto_arima(final_df[column], seasonal=False, suppress_warnings=True)
        print(f"Best ARIMA model for {column}: {model.order}")
    else:
        print(f"Column {column} not found in the DataFrame.")





                                  #models


        ## only for ARIMA AND SARIMA ##
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

# Assuming df is your stationary dataframe

# Dictionary of p, d, q values for each kit item
params = {
    'KIT0000896': (4, 0, 1),
    'KIT0000897': (0, 0, 0),
    'KIT0000642': (1, 0, 0),
    'KIT0000645': (0, 0, 0),
    'KIT0000647': (0, 0, 1),
    'KIT0000737': (1, 0, 0),
    'KIT0000285': (0, 0, 0),
    'KIT0000286': (1, 0, 0),
    'KIT0000949': (0, 0, 0),
    'KIT0000937': (0, 0, 0),
    'KIT0000938': (1, 0, 2),
    'KIT0000894': (0, 0, 0),
    'KIT0000895': (2, 0, 0),
    'KIT0000219': (3, 0, 1),
    'KIT0000933': (0, 0, 0),
    'KIT0000627': (0, 0, 0),
    'KIT0000581': (1, 0, 0),
    'KIT0000928': (0, 0, 0),
    'KIT0000365': (0, 0, 1),
    'KIT0000366': (2, 0, 0),
    'KIT0001042': (0, 0, 1),
    'KIT0000370': (0, 0, 0),
    'KIT0000445': (3, 0, 0),
    'KIT0000609': (0, 0, 1),
    'KIT0000905': (2, 0, 0),
    'KIT0000968': (0, 0, 1),
    'KIT0000468': (0, 0, 1),
    'KIT0000524': (0, 0, 0),
    'KIT0000563': (0, 0, 1),
    'KIT0000564': (1, 0, 0),
    'KIT0000565': (1, 0, 1),
    'KIT0000569': (0, 0, 0),
    'KIT0000994': (0, 0, 1),
    'KIT0000995': (0, 0, 1),
    'KIT0000975': (1, 0, 0),
    'KIT0000976': (0, 0, 0),
    'KIT0000978': (0, 0, 1),
    'KIT0001031': (2, 0, 0),
    'KIT0000417': (2, 0, 1),
    'KIT0000143': (0, 0, 1),
    'KIT0000144': (0, 0, 1),
    'KIT0000145': (0, 1, 1),
    'KIT0000146': (0, 0, 1),
    'KIT0000660': (0, 0, 1),
    'KIT0000868': (0, 0, 1),
    'KIT0000519': (0, 0, 1),
    'KIT0001030': (0, 0, 1),
    'KIT0000803': (0, 0, 0),
    'KIT0000854': (0, 0, 0),
    'KIT0001014': (0, 0, 1),
    'KIT0000467': (1, 0, 1),
    'KIT0000696': (0, 0, 0),
    'KIT0000170': (0, 0, 1),
    'KIT0001027': (0, 0, 1),
    'KIT0001028': (0, 0, 1),
    'KIT0000432': (0, 0, 1),
    'KIT0000860': (0, 0, 1),
    'KIT0000008': (0, 0, 1),
    'KIT0000010': (1, 0, 0),
    'KIT0000012': (0, 0, 0),
    'KIT0000013': (0, 0, 3),
    'KIT0000194': (0, 0, 1),
    'KIT0000195': (2, 0, 2),
    'KIT0000948': (0, 0, 1),
    'KIT0000953': (0, 0, 0),
    'KIT0000958': (0, 1, 0),
    'KIT0000985': (1, 0, 0),
    'KIT0001013': (0, 0, 0),
    'KIT0000582': (0, 0, 0),
    'KIT0001024': (0, 0, 1),
    'KIT0001017': (1, 0, 1),
    'KIT0000437': (0, 0, 0),
    'KIT0000296': (0, 0, 2),
    'KIT0000297': (0, 0, 1),
    'KIT0000355': (1, 0, 0),
    'KIT0000503': (0, 0, 1),
    'KIT0000203': (0, 0, 1),
    'KIT0000204': (1, 0, 1),
    'KIT0000460': (0, 0, 0),
    'KIT0000461': (0, 0, 2),
    'KIT0000959': (0, 0, 0),
    'KIT0000560': (2, 0, 0),
    'KIT0000990': (0, 0, 1),
    'KIT0000526': (0, 0, 1),
    'KIT0000294': (0, 0, 1),
    'KIT0000414': (1, 0, 1),
    'KIT0001037': (0, 0, 1),
    'KIT0000084': (0, 0, 1),
    'KIT0000174': (0, 0, 0),
    'KIT0000175': (1, 0, 0),
    'KIT0000227': (0, 0, 0),
    'KIT0000228': (1, 0, 0),
    'KIT0000353': (1, 0, 0),
    'KIT0000784': (0, 1, 1),
    'KIT0000016': (0, 0, 1),
    'KIT0000123': (2, 0, 1),
    'KIT0000147': (1, 0, 0),
    'KIT0000177': (0, 0, 0),
    'KIT0000256': (0, 0, 0),
    'KIT0000298': (0, 0, 1),
    'KIT0000942': (0, 0, 1),
    'KIT0001002': (0, 0, 0),
    'KIT0001004': (0, 0, 0),
    'KIT0001008': (0, 0, 0),
    'KIT0000162': (2, 0, 1),
    'KIT0000201': (0, 0, 2),
    'KIT0000884': (1, 0, 0),
    'KIT0000268': (0, 0, 0),
    'KIT0001000': (1, 0, 0),
    'KIT0000885': (0, 0, 1),
    'KIT0000886': (0, 0, 1),
    'KIT0000887': (2, 0, 2),
    'KIT0000888': (0, 0, 1),
    'KIT0001025': (0, 0, 1),
    'KIT0001026': (0, 0, 0)
    # Add more entries for other kit items
}

results = []

for kit_item, (p, d, q) in params.items():
    # Split the data into X and y
    y_train = final_df[kit_item].iloc[:-1]
    y_test = final_df[kit_item].iloc[-1]

    # Fit ARIMA model
    try:
        arima_model = ARIMA(y_train, order=(p, d, q))
        arima_fit = arima_model.fit()
    except:
        arima_fit = None

    # Fit SARIMA model
    try:
        sarima_model = SARIMAX(y_train, order=(p, d, q), seasonal_order=(1, 1, 0, 12))
        sarima_fit = sarima_model.fit()
    except:
        sarima_fit = None

    # Evaluate ARIMA model
    if arima_fit is not None:
        arima_pred = arima_fit.forecast(steps=1)
        arima_mape = mean_absolute_percentage_error([y_test], [arima_pred])
    else:
        arima_mape = None

    # Evaluate SARIMA model
    if sarima_fit is not None:
        sarima_pred = sarima_fit.forecast(steps=1)
        sarima_mape = mean_absolute_percentage_error([y_test], [sarima_pred])
    else:
        sarima_mape = None

    # Save the model if MAPE is more than 2%
    if arima_mape is not None and arima_mape > 2:
        joblib.dump(arima_fit, f"{kit_item}_arima_model.pkl")
    if sarima_mape is not None and sarima_mape > 2:
        joblib.dump(sarima_fit, f"{kit_item}_sarima_model.pkl")

    # Append the result to the list
    results.append({'Kit Item': kit_item, 'ARIMA MAPE': arima_mape, 'SARIMA MAPE': sarima_mape})

# Create a DataFrame from the results list
resultas_df = pd.DataFrame(results)

# Save the results to a CSV file
resultas_df.to_csv('model_evaluation_results.csv', index=False)



    ##  applying otrher models  ##


import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

# Assuming df is your stationary dataframe
X_train = final_df.iloc[:-1]
X_test =final_df.iloc[-1]

# Dictionary of p, d, q values for each kit item
params = {
    'KIT0000896': (4, 0, 1),
    'KIT0000897': (0, 0, 0),
    'KIT0000642': (1, 0, 0),
    'KIT0000645': (0, 0, 0),
    'KIT0000647': (0, 0, 1),
    'KIT0000737': (1, 0, 0),
    'KIT0000285': (0, 0, 0),
    'KIT0000286': (1, 0, 0),
    'KIT0000949': (0, 0, 0),
    'KIT0000937': (0, 0, 0),
    'KIT0000938': (1, 0, 2),
    'KIT0000894': (0, 0, 0),
    'KIT0000895': (2, 0, 0),
    'KIT0000219': (3, 0, 1),
    'KIT0000933': (0, 0, 0),
    'KIT0000627': (0, 0, 0),
    'KIT0000581': (1, 0, 0),
    'KIT0000928': (0, 0, 0),
    'KIT0000365': (0, 0, 1),
    'KIT0000366': (2, 0, 0),
    'KIT0001042': (0, 0, 1),
    'KIT0000370': (0, 0, 0),
    'KIT0000445': (3, 0, 0),
    'KIT0000609': (0, 0, 1),
    'KIT0000905': (2, 0, 0),
    'KIT0000968': (0, 0, 1),
    'KIT0000468': (0, 0, 1),
    'KIT0000524': (0, 0, 0),
    'KIT0000563': (0, 0, 1),
    'KIT0000564': (1, 0, 0),
    'KIT0000565': (1, 0, 1),
    'KIT0000569': (0, 0, 0),
    'KIT0000994': (0, 0, 1),
    'KIT0000995': (0, 0, 1),
    'KIT0000975': (1, 0, 0),
    'KIT0000976': (0, 0, 0),
    'KIT0000978': (0, 0, 1),
    'KIT0001031': (2, 0, 0),
    'KIT0000417': (2, 0, 1),
    'KIT0000143': (0, 0, 1),
    'KIT0000144': (0, 0, 1),
    'KIT0000145': (0, 1, 1),
    'KIT0000146': (0, 0, 1),
    'KIT0000660': (0, 0, 1),
    'KIT0000868': (0, 0, 1),
    'KIT0000519': (0, 0, 1),
    'KIT0001030': (0, 0, 1),
    'KIT0000803': (0, 0, 0),
    'KIT0000854': (0, 0, 0),
    'KIT0001014': (0, 0, 1),
    'KIT0000467': (1, 0, 1),
    'KIT0000696': (0, 0, 0),
    'KIT0000170': (0, 0, 1),
    'KIT0001027': (0, 0, 1),
    'KIT0001028': (0, 0, 1),
    'KIT0000432': (0, 0, 1),
    'KIT0000860': (0, 0, 1),
    'KIT0000008': (0, 0, 1),
    'KIT0000010': (1, 0, 0),
    'KIT0000012': (0, 0, 0),
    'KIT0000013': (0, 0, 3),
    'KIT0000194': (0, 0, 1),
    'KIT0000195': (2, 0, 2),
    'KIT0000948': (0, 0, 1),
    'KIT0000953': (0, 0, 0),
    'KIT0000958': (0, 1, 0),
    'KIT0000985': (1, 0, 0),
    'KIT0001013': (0, 0, 0),
    'KIT0000582': (0, 0, 0),
    'KIT0001024': (0, 0, 1),
    'KIT0001017': (1, 0, 1),
    'KIT0000437': (0, 0, 0),
    'KIT0000296': (0, 0, 2),
    'KIT0000297': (0, 0, 1),
    'KIT0000355': (1, 0, 0),
    'KIT0000503': (0, 0, 1),
    'KIT0000203': (0, 0, 1),
    'KIT0000204': (1, 0, 1),
    'KIT0000460': (0, 0, 0),
    'KIT0000461': (0, 0, 2),
    'KIT0000959': (0, 0, 0),
    'KIT0000560': (2, 0, 0),
    'KIT0000990': (0, 0, 1),
    'KIT0000526': (0, 0, 1),
    'KIT0000294': (0, 0, 1),
    'KIT0000414': (1, 0, 1),
    'KIT0001037': (0, 0, 1),
    'KIT0000084': (0, 0, 1),
    'KIT0000174': (0, 0, 0),
    'KIT0000175': (1, 0, 0),
    'KIT0000227': (0, 0, 0),
    'KIT0000228': (1, 0, 0),
    'KIT0000353': (1, 0, 0),
    'KIT0000784': (0, 1, 1),
    'KIT0000016': (0, 0, 1),
    'KIT0000123': (2, 0, 1),
    'KIT0000147': (1, 0, 0),
    'KIT0000177': (0, 0, 0),
    'KIT0000256': (0, 0, 0),
    'KIT0000298': (0, 0, 1),
    'KIT0000942': (0, 0, 1),
    'KIT0001002': (0, 0, 0),
    'KIT0001004': (0, 0, 0),
    'KIT0001008': (0, 0, 0),
    'KIT0000162': (2, 0, 1),
    'KIT0000201': (0, 0, 2),
    'KIT0000884': (1, 0, 0),
    'KIT0000268': (0, 0, 0),
    'KIT0001000': (1, 0, 0),
    'KIT0000885': (0, 0, 1),
    'KIT0000886': (0, 0, 1),
    'KIT0000887': (2, 0, 2),
    'KIT0000888': (0, 0, 1),
    'KIT0001025': (0, 0, 1),
    'KIT0001026': (0, 0, 0)
}

results = []

for kit_item, (p, d, q) in params.items():
    # Split the data into X and y
    y_train = X_train[kit_item]
    X_train_kit = np.arange(len(y_train)).reshape(-1, 1)
    
    y_test = X_test[kit_item]
    X_test_kit = np.array([[len(y_train)]]).reshape(-1, 1)

    # Initialize models
    models = [
        ('Linear Regression', LinearRegression()),
        ('Random Forest', RandomForestRegressor()),
        ('XGBoost', XGBRegressor())
    ]

    best_model = None
    best_mape = float('inf')

    for model_name, model in models:
        # Train the model
        model.fit(X_train_kit, y_train)

        # Make predictions
        y_pred = model.predict(X_test_kit)

        # Calculate MAPE
        mape = mean_absolute_percentage_error([y_test], [y_pred])

        # Update best model if MAPE is lower
        if mape < best_mape:
            best_mape = mape
            best_model = model

    # Save the best model if MAPE is less than 20%
    if best_mape < 20:
        joblib.dump(best_model, f"{kit_item}_model.pkl")

    # Append the result to the list
    results.append({'Kit Item': kit_item, 'Best Model': type(best_model).__name__, 'MAPE': best_mape})

# Create a DataFrame from the results list
rest_results_df = pd.DataFrame(results)

# Save the results to a CSV file
rest_results_df.to_csv('model_evaluation_results_second.csv', index=False)












import pandas as pd

# Load the CSV files into DataFrames
arima_sarima_df = pd.read_csv('model_evaluation_results.csv')
ml_models_df = pd.read_csv('model_evaluation_results_second.csv')

# Merge the DataFrames on the 'kititem' column
merged_df = pd.merge(arima_sarima_df, ml_models_df, on='Kit Item', suffixes=('_arima_sarima', '_ml_models'))

# Select the best model for each kit item based on the lowest MAPE value
merged_df['Best Model'] = merged_df[['MAPE Value_arima_sarima', 'MAPE Value_ml_models']].idxmin(axis=1).str.replace('MAPE Value_', '')

# Create a new DataFrame with the 'kititem', 'Best Model', and 'MAPE Value' columns
result_df = merged_df[['Kit Item', 'Best Model', 'MAPE Value_' + merged_df['Best Model']]]

# Save the new DataFrame to a new CSV file
result_df.to_csv('best_models_mape.csv', index=False)

## Extracted final output file ##

# Merge the DataFrames on the 'Kit Item' column
merged_df = pd.merge(file1_df, file2_df, on='Kit Item')
# Compare the MAPE values and update other columns
merged_df['Best Model'] = merged_df.apply(lambda row: row['Best Model_x'] if row['MAPE_x'] < row['MAPE_y'] else row['Best Model_y'], axis=1)
merged_df['Updated Column'] = merged_df.apply(lambda row: row['MAPE_x'] if row['MAPE_x'] < row['MAPE_y'] else row['MAPE_y'], axis=1)

# Print the MAPE value in the 'Updated Column'
merged_df['Updated MAPE'] = merged_df.apply(lambda row: row['MAPE_x'] if row['MAPE_x'] < row['MAPE_y'] else row['MAPE_y'], axis=1)

# Save the updated DataFrame to a new CSV file
merged_df.to_csv('outputfinal.csv', index=False)




































                                        ### Rough Work ###



    ##### ACF and PACF plot to find values of 'P'and 'Q' ######

from pmdarima import auto_arima

# Filter columns that start with "KIT"
kit_columns = [col for col in stationary_df.columns if col.startswith('KIT')]

 ## To get p,d,q values using autoarima ##
# Fit AutoARIMA model for each column

for column in kit_columns:
    if column in stationary_df.columns:
        model = auto_arima(stationary_df[column], seasonal=False, suppress_warnings=True)
        print(f"Best ARIMA model for {column}: {model.order}")
    else:
        print(f"Column {column} not found in the DataFrame.")






                           
                ### model building Using ARIMA ###

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Split the data into training and test sets
train_size = int(len(stationary_df) * 0.8)  # Use 80% of the data for training
train_data = stationary_df.iloc[:train_size]
test_data = stationary_df.iloc[train_size:]

# Dictionary mapping column names to (p, d, q) values
arima_orders = {
    'KIT0000896': (3, 0, 1),
    'KIT0000897': (0, 0, 0),
    'KIT0000642': (0, 0, 0),
    'KIT0000645': (0, 0, 0),
    'KIT0000647': (0, 0, 0),
    'KIT0000737': (1, 0, 0),
    'KIT0000286': (1, 0, 0),
    'KIT0000937': (0, 0, 0),
    'KIT0000938': (0, 0, 0),
    'KIT0000894': (0, 0, 0),
    'KIT0000219': (3, 0, 1),
    'KIT0000933': (0, 0, 0),
    'KIT0000627': (2, 0, 3),
    'KIT0000581': (0, 0, 0),
    'KIT0000370': (0, 0, 0),
    'KIT0000524': (0, 0, 0),
    'KIT0000564': (1, 0, 0),
    'KIT0000565': (1, 0, 0),
    'KIT0000569': (0, 0, 0),
    'KIT0000975': (1, 0, 0),
    'KIT0000976': (0, 0, 0),
    'KIT0001031': (3, 0, 0),
    'KIT0000145': (0, 0, 0),
    'KIT0001030': (0, 0, 0),
    'KIT0000803': (0, 0, 0),
    'KIT0000854': (1, 0, 0),
    'KIT0000696': (0, 0, 0),
    'KIT0001028': (0, 0, 0),
    'KIT0000008': (0, 0, 1),
    'KIT0000010': (1, 0, 0),
    'KIT0000012': (0, 0, 1),
    'KIT0000013': (0, 0, 3),
    'KIT0000195': (3, 0, 2),
    'KIT0000958': (2, 0, 1),
    'KIT0001017': (1, 0, 1),
    'KIT0000355': (1, 0, 0),
    'KIT0000203': (1, 0, 0),
    'KIT0000460': (0, 0, 0),
    'KIT0000461': (0, 0, 0),
    'KIT0000959': (0, 0, 0),
    'KIT0000526': (0, 0, 1),
    'KIT0000084': (0, 0, 1),
    'KIT0000174': (0, 0, 0),
    'KIT0000175': (1, 0, 0),
    'KIT0000227': (0, 0, 0),
    'KIT0000228': (1, 0, 0),
    'KIT0000784': (1, 0, 0),
    'KIT0000016': (0, 0, 1),
    'KIT0000147': (1, 0, 0),
    'KIT0000177': (0, 0, 0),
    'KIT0000256': (0, 0, 0),
    'KIT0001002': (0, 0, 0),
    'KIT0001004': (0, 0, 0),
    'KIT0001008': (0, 0, 0),
    'KIT0000162': (0, 0, 1),
    'KIT0001000': (1, 0, 0),
    'KIT0001026': (0, 0, 0)
}
# Iterate over each column and its corresponding (p, d, q) values
for column, order in arima_orders.items():
    if column in stationary_df.columns:
        # Fit ARIMA model
        model = ARIMA(train_data[column], order=order, freq='MS')
        fitted_model = model.fit()

        # Make predictions
        start_index = len(train_data[column])
        end_index = start_index + len(test_data[column]) - 1
        predictions = fitted_model.predict(start=start_index, end=end_index)

        # Calculate RMSE
        actual_values = test_data[column]
        rmse = np.sqrt(mean_squared_error(actual_values, predictions))

        print(f"RMSE for {column}: {rmse}")
    else:
        print(f"Column {column} not found in the DataFrame.") 




    ##### for non_stationary data_frame #####


import numpy as np
from pmdarima import auto_arima

# Assuming 'non_stationary_df' is your DataFrame
data_1d = non_stationary_df.values.flatten()

# Use auto_arima with the 1D array
model = auto_arima(data_1d, d=1, seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore")


import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Differencing to make the data stationary
conv_stationary_df = non_stationary_df.diff().dropna()







### Applying Auto arima to get p,d,q from non_stationarty data ###

from pmdarima import auto_arima

# Initialize an empty dictionary to store the results for each column
arima_results = {}

# Loop through each column in the non-stationary DataFrame
for col in conv_stationary_df.columns:
    # Fit an ARIMA model using auto_arima for the current column
    model = auto_arima(conv_stationary_df[col], seasonal=False, suppress_warnings=True)
    
    # Store the order (p,d,q) values in the dictionary
    arima_results[col] = model.order

# Display the order (p,d,q) values for each column
for col, order in arima_results.items():
    print(f"Column: {col}, Order: {order}")




from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Dictionary of kit items and their respective (p, d, q) values
kit_items_orders = {
    'KIT0000285': (0, 0, 0),
    'KIT0000949': (0, 0, 1),
    'KIT0000895': (0, 0, 1),
    'KIT0000928': (0, 0, 0),
    'KIT0000365': (0, 0, 1),
    'KIT0000366': (2, 0, 0),
    'KIT0001042': (0, 0, 1),
    'KIT0000445': (3, 0, 0),
    'KIT0000609': (1, 0, 0),
    'KIT0000905': (2, 0, 0),
    'KIT0000968': (0, 0, 1),
    'KIT0000468': (0, 0, 1),
    'KIT0000563': (0, 0, 1),
    'KIT0000994': (0, 0, 1),
    'KIT0000995': (0, 0, 1),
    'KIT0000978': (0, 0, 1),
    'KIT0000417': (2, 0, 2),
    'KIT0000143': (0, 0, 1),
    'KIT0000144': (0, 0, 1),
    'KIT0000146': (0, 0, 1),
    'KIT0000660': (0, 0, 1),    
    'KIT0000868': (0, 0, 1),
    'KIT0000519': (0, 0, 1),
    'KIT0001014': (0, 0, 1),
    'KIT0000467': (1, 0, 1),
    'KIT0000170': (0, 0, 1),
    'KIT0001027': (0, 0, 1),
    'KIT0000432': (0, 0, 1),
    'KIT0000860': (0, 0, 2),
    'KIT0000194': (0, 0, 1),
    'KIT0000948': (0, 0, 1),
    'KIT0000953': (0, 0, 0),
    'KIT0000985': (0, 0, 1),
    'KIT0001013': (0, 0, 0),
    'KIT0000582': (0, 0, 0),
    'KIT0001024': (0, 0, 1),
    'KIT0000437': (0, 0, 1),
    'KIT0000296': (0, 0, 2),
    'KIT0000297': (1, 0, 1),
    'KIT0000503': (0, 0, 1),
    'KIT0000204': (0, 0, 1),
    'KIT0000560': (2, 0, 0),
    'KIT0000990': (0, 0, 1),
    'KIT0000294': (0, 0, 1),
    'KIT0000414': (0, 0, 1),
    'KIT0001037': (0, 0, 1),
    'KIT0000353': (1, 0, 0),
    'KIT0000123': (0, 0, 1),
    'KIT0000298': (0, 0, 1),
    'KIT0000942': (0, 0, 1),
    'KIT0000201': (0, 0, 1),
    'KIT0000884': (0, 0, 1),
    'KIT0000268': (0, 0, 1),
    'KIT0000885': (0, 0, 1),
    'KIT0000886': (0, 0, 1),
    'KIT0000887': (2, 0, 2),
    'KIT0000888': (0, 0, 1),
    'KIT0001025': (0, 0, 1)

}

# Initialize dictionaries to store RMSE and MAPE values
rmse_values = {}
mape_values = {}

# Build ARIMA models and calculate RMSE and MAPE for each kit item
for kit_item, order in kit_items_orders.items():
    # Fit ARIMA model
    model = ARIMA(non_stationary_df[kit_item], order=order)
    fitted_model = model.fit()

    # Forecast
    forecast = fitted_model.forecast(steps=len(test))

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test[kit_item], forecast))
    rmse_values[kit_item] = rmse

    # Calculate MAPE
    mape = np.mean(np.abs((test[kit_item] - forecast) / test[kit_item])) * 100
    mape_values[kit_item] = mape

    # Print RMSE and MAPE values for each kit item
    print(f"Kit Item: {kit_item}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}")

# Calculate overall RMSE and MAPE
overall_rmse = np.mean(list(rmse_values.values()))
overall_mape = np.mean(list(mape_values.values()))
print(f"Overall RMSE: {overall_rmse:.2f}, Overall MAPE: {overall_mape:.2f}")



        ### trial 1 ###

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split

# Dictionary of kit items and their respective (p, d, q) values
kit_items_orders = {
    'KIT0000285': (0, 0, 0),
    'KIT0000949': (0, 0, 1),
    'KIT0000895': (0, 0, 1),
    'KIT0000928': (0, 0, 0),
    'KIT0000365': (0, 0, 1),
    'KIT0000366': (2, 0, 0),
    'KIT0001042': (0, 0, 1),
    'KIT0000445': (3, 0, 0),
    'KIT0000609': (1, 0, 0),
    'KIT0000905': (2, 0, 0),
    'KIT0000968': (0, 0, 1),
    'KIT0000468': (0, 0, 1),
    'KIT0000563': (0, 0, 1),
    'KIT0000994': (0, 0, 1),
    'KIT0000995': (0, 0, 1),
    'KIT0000978': (0, 0, 1),
    'KIT0000417': (2, 0, 2),
    'KIT0000143': (0, 0, 1),
    'KIT0000144': (0, 0, 1),
    'KIT0000146': (0, 0, 1),
    'KIT0000660': (0, 0, 1),    
    'KIT0000868': (0, 0, 1),
    'KIT0000519': (0, 0, 1),
    'KIT0001014': (0, 0, 1),
    'KIT0000467': (1, 0, 1),
    'KIT0000170': (0, 0, 1),
    'KIT0001027': (0, 0, 1),
    'KIT0000432': (0, 0, 1),
    'KIT0000860': (0, 0, 2),
    'KIT0000194': (0, 0, 1),
    'KIT0000948': (0, 0, 1),
    'KIT0000953': (0, 0, 0),
    'KIT0000985': (0, 0, 1),
    'KIT0001013': (0, 0, 0),
    'KIT0000582': (0, 0, 0),
    'KIT0001024': (0, 0, 1),
    'KIT0000437': (0, 0, 1),
    'KIT0000296': (0, 0, 2),
    'KIT0000297': (1, 0, 1),
    'KIT0000503': (0, 0, 1),
    'KIT0000204': (0, 0, 1),
    'KIT0000560': (2, 0, 0),
    'KIT0000990': (0, 0, 1),
    'KIT0000294': (0, 0, 1),
    'KIT0000414': (0, 0, 1),
    'KIT0001037': (0, 0, 1),
    'KIT0000353': (1, 0, 0),
    'KIT0000123': (0, 0, 1),
    'KIT0000298': (0, 0, 1),
    'KIT0000942': (0, 0, 1),
    'KIT0000201': (0, 0, 1),
    'KIT0000884': (0, 0, 1),
    'KIT0000268': (0, 0, 1),
    'KIT0000885': (0, 0, 1),
    'KIT0000886': (0, 0, 1),
    'KIT0000887': (2, 0, 2),
    'KIT0000888': (0, 0, 1),
    'KIT0001025': (0, 0, 1)

}

# Initialize dictionaries to store RMSE and MAPE values
rmse_values = {}
mape_values = {}

# Split data into training and test sets
train_size = int(0.8 * len(non_stationary_df))
train, test = non_stationary_df.iloc[:train_size], non_stationary_df.iloc[train_size:]

# Build ARIMA models and calculate RMSE and MAPE for each kit item
for kit_item, order in kit_items_orders.items():
    # Fit ARIMA model
    model = ARIMA(train[kit_item], order=order)
    fitted_model = model.fit()

    # Forecast
    forecast = fitted_model.forecast(steps=len(test))

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test[kit_item], forecast))
    rmse_values[kit_item] = rmse

    # Calculate MAPE
    mape = np.mean(np.abs((test[kit_item] - forecast) / test[kit_item])) * 100
    mape_values[kit_item] = mape

    # Print RMSE and MAPE values for each kit item
    print(f"Kit Item: {kit_item}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}")

# Calculate overall RMSE and MAPE
overall_rmse = np.mean(list(rmse_values.values()))
overall_mape = np.mean(list(mape_values.values()))
print(f"Overall RMSE: {overall_rmse:.2f}, Overall MAPE: {overall_mape:.2f}")


import numpy as np

# Calculate MAPE
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Actual values of the time series
actual_values = conv_stationary_df.iloc[-forecast_steps:]

# Forecasted values from the ARIMA model
forecast_values = forecast

# Calculate MAPE
mape = calculate_mape(actual_values, forecast_values)

print(f"MAPE: {mape:.2f}%")

















# Step 3: Build an ARIMA model
# Example: ARIMA(1, 0, 1) for demonstration purposes
model = ARIMA(conv_stationary_df, order=(1, 0, 1))
fitted_model = model.fit()

# Step 4: Validate the model
# Split the data into training and test sets
train_size = int(len(stationary_df) * 0.8)
train_data = stationary_df.iloc[:train_size]
test_data = stationary_df.iloc[train_size:]

# Fit the model on the training data
fitted_model = model.fit()

# Make predictions
predictions = fitted_model.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Step 5: Evaluate the model
rmse = np.sqrt(mean_squared_error(test_data, predictions))
print(f"RMSE: {rmse}")

# Step 6: Make predictions for future time periods
future_predictions = fitted_model.forecast(steps=5)  # Forecast the next 5 time periods

# You can then inverse the differencing to get the predictions in the original scale if needed























































































































        ### visualizing stationary data ###
import matplotlib.pyplot as plt

# Plotting all stationary columns
fig, ax = plt.subplots(figsize=(12, 8))
for column in stationary_df.columns:
    ax.plot(stationary_df[column], label=column)
ax.set_title('Stationary Time Series')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.show()


  ### visualizing non_stationary data ###
import matplotlib.pyplot as plt

# Plotting all non-stationary columns
fig, ax = plt.subplots(figsize=(12, 8))
for column in non_stationary_df.columns:
    ax.plot(non_stationary_df[column], label=column)
ax.set_title('Non-Stationary Time Series')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.show()



            ###### MODEL BUILDING #######

# 1.importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 2.spliting the data , 80% for training, 20% for testing

train_size = int(0.8 * len(stationary_df))  
train, test = stationary_df.iloc[:train_size], stationary_df.iloc[train_size:]

# Example: ARIMA(1, 0, 1)

### not getting that p,d,q values.......

p, d, q = 1, 0, 1
model = ARIMA(train, order=(p, d, q))
fitted_model = model.fit()

predictions = fitted_model.predict(start=test.index[0], end=test.index[-1])


## evaluating  model ###
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


##### Ploting #####
plt.figure(figsize=(12, 8))
plt.plot(train, label='Training Data')
plt.plot(test.index, test, label='Actual Data')
plt.plot(test.index, predictions, label='Predictions')
plt.title('ARIMA Model Predictions')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()























































transposed_df1 = transposed_df1.drop('index', axis=1)

transposed_df1.reset_index(drop=True, inplace=True)
# Assuming your DataFrame is named df
transposed_df1.index = transposed_df1.columns

# Assuming df is your DataFrame
transposed_df1.iloc[0] = transposed_df1.iloc[1]

df2 = transposed_df1.set_axis('KIT0000896', 'KIT0000897', 'KIT0000613', 'KIT0000634', 'KIT0000638', 'KIT0000642', 'KIT0000645', 'KIT0000646', 'KIT0000647', 'KIT0000725', 'KIT0000735', 'KIT0000737', 'KIT0000738', 'KIT0000741', 'KIT0000792', 'KIT0000813', 'KIT0000815', 'KIT0000285', 'KIT0000286', 'KIT0000949', 'KIT0001052', 'KIT0001053', 'KIT0001054', 'KIT0000937', 'KIT0000938', 'KIT0000939', 'KIT0001016', 'KIT0001038', 'KIT0001039', 'KIT0000894', 'KIT0000895', 'KIT0001157', 'KIT0001158', 'KIT0001134', 'KIT0001135', 'KIT0001136', 'KIT0001137', 'KIT0001151', 'KIT0001152', 'KIT0001153', 'KIT0000219', 'KIT0001061', 'KIT0000933', 'KIT0000627', 'KIT0001066', 'KIT0000580', 'KIT0000581', 'KIT0000862', 'KIT0000928', 'KIT0000365', 'KIT0000366', 'KIT0001042', 'KIT0000370', 'KIT0000445', 'KIT0000609', 'KIT0000905', 'KIT0000968', 'KIT0000468', 'KIT0000524', 'KIT0000563', 'KIT0000564', 'KIT0000565', 'KIT0000566', 'KIT0000569', 'KIT0000877', 'KIT0000994', 'KIT0000995', 'KIT0000996', 'KIT0001012', 'KIT0001158', 'KIT0000975', 'KIT0000976', 'KIT0000977', 'KIT0000978', 'KIT0001107', 'KIT0001108', 'KIT0001134', 'KIT0001135', 'KIT0001136', 'KIT0001137', 'KIT0001151', 'KIT0001152', 'KIT0001153', 'KIT0000199', 'KIT0000783', 'KIT0001018', 'KIT0001019', 'KIT0001020', 'KIT0001021', 'KIT0001022', 'KIT0001029', 'KIT0001031', 'KIT0001065', 'KIT0001112', 'KIT0001121', 'KIT0001139', 'KIT0001145', 'KIT0001146', 'KIT0001147', 'KIT0001046', 'KIT0000024', 'KIT0000029', 'KIT0000030', 'KIT0000032', 'KIT0000417', 'KIT0000997', 'KIT0000143', 'KIT0000144', 'KIT0000145', 'KIT0000146', 'KIT0000660', 'KIT0001140', 'KIT0000868', 'KIT0001109', 'KIT0000519', 'KIT0001102', 'KIT0001030', 'KIT0000803', 'KIT0000746', 'KIT0000748', 'KIT0000749', 'KIT0000750', 'KIT0000805', 'KIT0000864', 'KIT0000865', 'KIT0000866', 'KIT0000867', 'KIT0000869', 'KIT0000870', 'KIT0000871', 'KIT0001063', 'KIT0000836', 'KIT0001103', 'KIT0000854', 'KIT0000855', 'KIT0001150', 'KIT0001154', 'KIT0001014', 'KIT0000608', 'KIT0000212', 'KIT0001050', 'KIT0001051', 'KIT0001156', 'KIT0001159', 'KIT0001160', 'KIT0000467', 'KIT0000696', 'KIT0000943', 'KIT0001118', 'KIT0001119', 'KIT0000170', 'KIT0001027', 'KIT0001028', 'KIT0001093', 'KIT0001097', 'KIT0001098', 'KIT0000432', 'KIT0000860', 'KIT0001138', 'KIT0000004', 'KIT0000008', 'KIT0000010', 'KIT0000012', 'KIT0000013', 'KIT0000194', 'KIT0000195', 'KIT0000712', 'KIT0000948', 'KIT000095', axis=1)


df2 = transposed_df1.iloc[1:]

# Reset the index
df2 = transposed_df1.reset_index(drop=True)
   
df2 = transposed_df1.drop(columns=['KIT ITEM'])




use iloc to drop






# COMBINE REQUIRED DATAFRAME
final_df = pd.concat([transposed_df1, transposed_df])

# Display the concatenated DataFrame
print(final_df)


final_df = pd.concat([transposed_df1, transposed_df])

# Resetting index and dropping the original index column
lst_df = final_df.reset_index(drop=True)

# Display the concatenated DataFrame without index row and column
print(lst_df)

lst_df = lst_df.set_index('')









req1=df[req]
print (req1)
 
df = req1.transpose()
df.reaq1 = df.iloc[0]
df = df[1:]
df



req = ['KIT ITEM','4/1/2021 0:00','5/1/2021 0:00','6/1/2021 0:00','7/1/2021 0:00','8/1/2021 0:00','9/1/2021 0:00','10/1/2021 0:00','11/1/2021 0:00','12/1/2021 0:00','1/1/2022 0:00','2/1/2022 0:00','3/1/2022 0:00','4/1/2022 0:00','5/1/2022 0:00','6/1/2022 0:00','7/1/2022 0:00','8/1/2022 0:00','9/1/2022 0:00','10/1/2022 0:00','11/1/2022 0:00','12/1/2022 0:00','1/1/2023 0:00','2/1/2023 0:00','3/1/2023 0:00','4/1/2023 0:00','5/1/2023 0:00','6/1/2023 0:00','7/1/2023 0:00','8/1/2023 0:00','9/1/2023 0:00','10/1/2023 0:00','11/1/2023 0:00','12/1/2023 0:00'
]


df1 = ['KIT ITEM'];























# Given column names
columns = [
    '4/1/2021 0:00', '5/1/2021 0:00', '6/1/2021 0:00', '7/1/2021 0:00', '8/1/2021 0:00', '9/1/2021 0:00',
    '10/1/2021 0:00', '11/1/2021 0:00', '12/1/2021 0:00', '1/1/2022 0:00', '2/1/2022 0:00', '3/1/2022 0:00',
    '4/1/2022 0:00', '5/1/2022 0:00', '6/1/2022 0:00', '7/1/2022 0:00', '8/1/2022 0:00', '9/1/2022 0:00',
    '10/1/2022 0:00', '11/1/2022 0:00', '12/1/2022 0:00', '1/1/2023 0:00', '2/1/2023 0:00', '3/1/2023 0:00',
    '4/1/2023 0:00', '5/1/2023 0:00', '6/1/2023 0:00', '7/1/2023 0:00', '8/1/2023 0:00', '9/1/2023 0:00',
    '10/1/2023 0:00', '11/1/2023 0:00', '12/1/2023 0:00'
]

# Create a DataFrame with the given column names
df = pd.DataFrame(columns, columns=['Dates'])

# Transpose the DataFrame and set the first row as the index
df = df.transpose()
df.columns = df.iloc[0]
df = df[1:]

# Set a name for the index column
df.index.name = 'Date'

print(df)













req1.set_index('KIT ITEM',inplace=True)

req1

df = req1.transpose()
df

#to treat duplicate we do,
dup_clms=df.columns[df.columns.duplicated()].tolist()
dup_clms

#aggrigate duplicate
for col in dup_clms:
        df[col]=df[col].sum(axis=1)
        

df = df.loc[:,df.columns.duplicated()]

df.info() 


lst_data=df.isnull().sum() 


# Select specific columns


