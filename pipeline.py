#This pipeline takes in a csv file and returns relevant business information, closure predictions, and probabilities
import pandas as pd
import numpy as np
#import difflib #this will be used for fuzzy joins
#from fuzzywuzzy import fuzz
#from fuzzywuzzy import process
from math import radians, sin, cos, sqrt, asin
#from tqdm import tqdm
from sklearn.cluster import DBSCAN
import re
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

#This is the data we are generating predictions for
#claud4 = pd.read_csv("C:/Users/barba/OneDrive/Documents/MSBA2/IDS_560/fp_deliverable/claud4.csv")

#First processing of data to get it into join ready form
def parse_store_hours(row):
    if pd.isnull(row):
        return [None] * 7  # Return a list of None values if the row is missing
    hours_by_day = {str(i): None for i in range(1, 8)}  # Initialize days with None values
    for part in row.split('|'):
        # Check if ':' is in the part, indicating a day:hours format
        if ':' in part:
            day, text = part.split(': ', 1)  # Split on the first occurrence of ': '
            hours_by_day[day] = part  # Store the entire part, e.g., "1: 10AM–8PM"
        else:
            # If ':' is not present, assume the entire part is to be used as is
            # It could be just a day number or some other format; treated as an entire record
            hours_by_day[part] = part  # The key and value are the same because the format is undetermined
    return [hours_by_day[str(i)] for i in range(1, 8)]  # Return a list of records for each day

def processData(jimData, day_of_week = 'Monday', day_of_disaster = 4):
    #Extract store hours into their respective columns
    jimData[['WT1', 'WT2', 'WT3', 'WT4', 'WT5', 'WT6', 'WT7']] = pd.DataFrame(jimData['workTime'].apply(parse_store_hours).tolist(), index=jimData.index)
    
    #day_of_week = str(input("What is the day of the week?"))
    #day_of_disaster = int(input("What is the day of disaster?"))
    jimData["Day of Week"] = day_of_week
    jimData["Day of Disaster"] = day_of_disaster
    
    selected_columns = jimData[['Date', 'Disaster Name', 'Year', 'Day of Week', 'Day of Disaster', 'PID',
       'company_address2', 'LON', 'LAT', 'COMPANY_NAME', 'CATEGORY1', 'StoreType',
       'RATING', 'REVIEW_NUMBER', 'pics2', 'verified',
       'tempClosed', 'WT1', 'WT2', 'WT3', 'WT4', 'WT5', 'WT6', 'WT7']]
    

    return selected_columns.rename(columns={'company_address2': 'full_location'})

#procHist = processData(claud4)



def joinCensus(data):
    #Join the Census Data
    us_states = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming"
    }
    #Join the census data on

    #Extract elemenst of address for joining
    data["state"] = data['full_location'].str.extract(r',\s([A-Z]{2})\s')
    data['zip_code'] = data['full_location'].str.extract(r'(\d{5})$')
    data['county'] = data['full_location'].str.extract(r'^(.*?),')
    unique_states = data["state"].unique()

    #Census encodes each state as a number so you need the below file
    state_code_path = "data/state_codes.txt"
    state_codes = pd.read_csv(state_code_path, delimiter='|')
    state_dict = {}
    for key, value in zip(state_codes['STUSAB'], state_codes['STATE']):
        if (key in unique_states):
            state_dict[key] = value

    state_dict['AL'] = '01'
    state_dict['AK'] = '02'
    state_dict['AZ'] = '04'
    state_dict['AR'] = '05'
    state_dict['CA'] = '06'
    state_dict['CO'] = '08'
    state_dict['CT'] = '09'

    #Staging step for join
    file_path = 'data/new_census_google_mappings.csv'
    map_df = pd.read_csv(file_path)
    map_dict = dict(zip(map_df['Original'], map_df['Match']))
    #creates census category column using mapping table
    data["Census_Category"] = data['CATEGORY1'].map(map_dict)

    data["full_state"] = data["state"].map(us_states)
    #Need this column to do a fuzzy mapping of the google locations to census county locations
    data["Place and State"] = data["county"] + ', ' + data["full_state"]

    #reads in resulting df from last section 
    place_mappings = pd.read_csv('data/census_place_mappings.csv')
    census_df = pd.read_csv("data/census_data.csv")

    #This will join the whole table with clean. We are intresting in the 'Match' column
    final_clean = pd.merge(data, place_mappings, left_on="Place and State", right_on="Original", how='left')


    final_clean = final_clean.drop('Unnamed: 0', axis=1)
    final_clean = final_clean.drop('Score', axis=1)
    final_clean = final_clean.drop('Original', axis=1)

    final_clean = final_clean.rename(columns={'Match' : 'Census_Place'})

    #This is an important checkpoint to check for null values before performing final join
    joined_table = pd.merge(final_clean, census_df, left_on=['Census_Category', 'Census_Place'], right_on =['NAICS2017_LABEL', 'NAME'], how='left')
    joined_table = joined_table.drop(columns=['Unnamed: 0', 'NAICS2017_LABEL', 'NAME', 'GEO_ID','INDGROUP', 'INDLEVEL','SECTOR', 'SUBSECTOR', 'TAXSTAT', 'NAICS2017',
        'state_y', 'place', 'MINLEVEL', 'State Name'])

    return joined_table

#censusJoin = joinCensus(procHist)

#Join the FEMA Scores
def joinFema(data):
    #Join the FEMA Data on

    #This is the FEMA scores provided by the fall 2023 semester
    index_path = 'data/final_clean_data.xlsx'
    zip_to_county = pd.read_excel(index_path, sheet_name='Sheet5', dtype={'zip': str}) #mapping table
    index_data = pd.read_excel(index_path, sheet_name='SVI') #Actual data

    #extracts county name
    index_data['county_map'] = index_data['Key'].str.split(',').str[0]

    #merges provided tables from fall 2023
    merged_index = pd.merge(zip_to_county, index_data, left_on=['state', 'county'], right_on=['STATEABBRV', 'county_map'], how='left')
    merged_index = merged_index.drop(columns=['state', 'county', 'Key', 'NRI_ID', 'STATE', 'STATEABBRV', 'COUNTY', 'COUNTYTYPE','county_map'])

    #clean zip code
    data['zip_code'] = data['zip_code'].apply(lambda x: str(int(x)) if pd.notnull(x) else x)

    final = pd.merge(data, merged_index, left_on='zip_code', right_on='zip', how='left')
    final = final.drop(columns=['zip'])
    columns_to_inpute = ['pics2', 'verified', 'tempClosed']

    for column in columns_to_inpute:
        final[column] = final[column].fillna(0)

    #print(final.isnull().sum())
    
    #Dropping null values and duplicates because this demo assumes that the schedule and census data is available for all given records
    return final.dropna().drop_duplicates()

#femaJoin = joinFema(censusJoin)

#Adding engineered features besides GMM Clusters
def createFeatures(data):
    #Engineer Features
    
    #Get rid of duplicates
    grouped = data.groupby(['Date', 'Day of Disaster', 'Disaster Name', 'PID', 'COMPANY_NAME']).size()
    filtered_grouped = grouped[grouped > 1]
    indices_to_exclude = filtered_grouped.index
    filtered_df = data[~data.set_index(['Date', 'Day of Disaster', 'Disaster Name', 'PID', 'COMPANY_NAME']).index.isin(indices_to_exclude)]

    #This is the createion of the 24 hour and 7 days a week columns
    columns_to_check = ['WT1', 'WT2', 'WT3', 'WT4', 'WT5', 'WT6', 'WT7']
    filtered_df['24_hours'] = filtered_df.apply(lambda row: any('24' in str(row[col]) for col in columns_to_check), axis=1).astype(int)
    for col in columns_to_check:
        filtered_df[col] = (~filtered_df[col].isna()).astype(int)  # Convert non-NaN to 1 and NaN to 0
    filtered_df['7_days'] = (filtered_df[columns_to_check].sum(axis=1) == 7).astype(int)

    appHist = pd.read_csv("data/appHist.csv")
    filtered_df['distance'] = 1000
    common_columns = filtered_df.columns.intersection(appHist.columns)
    print(common_columns)

    combined_df = pd.concat([appHist[common_columns], filtered_df[common_columns]], ignore_index=True)
    
    #The creation of lagging variables for distance and tempClosed
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='mixed')
    combined_df = combined_df.sort_values(by=['Disaster Name',   'PID', 'COMPANY_NAME', 'Day of Disaster', 'Date'])

    combined_df = combined_df.sort_values(by=['Disaster Name',   'PID', 'COMPANY_NAME', 'Day of Disaster', 'Date'])

    combined_df = (combined_df
        .assign(
            tempClosed_lag1 = lambda x: x.groupby(['Disaster Name',   'PID', 'COMPANY_NAME'])['tempClosed'].shift(1),
            tempClosed_lag2 = lambda x: x.groupby(['Disaster Name',   'PID', 'COMPANY_NAME'])['tempClosed'].shift(2),
            tempClosed_lag3 = lambda x: x.groupby(['Disaster Name',   'PID', 'COMPANY_NAME'])['tempClosed'].shift(3),
            tempClosed_lag4 = lambda x: x.groupby(['Disaster Name',   'PID', 'COMPANY_NAME'])['tempClosed'].shift(4),
            tempClosed_lag5 = lambda x: x.groupby(['Disaster Name',   'PID', 'COMPANY_NAME'])['tempClosed'].shift(5),
        )
    )

    combined_df = combined_df.sort_values(by=['Disaster Name',   'PID', 'COMPANY_NAME', 'Day of Disaster', 'Date'])

    combined_df = (combined_df
        .assign(
            distance_lag1 = lambda x: x.groupby(['Disaster Name',   'PID', 'COMPANY_NAME'])['distance'].shift(1),
        )
    )

    curr_df = combined_df[combined_df['Day of Disaster'] == 4]
    curr_df['distance_lag1'] = curr_df['distance_lag1'].fillna(1000)


    
    return curr_df.fillna(0)

#modelData = createFeatures(femaJoin)

def manual_one_hot_encode(df, column_name, ohe_cols):
    """
    Manually one-hot encodes the specified column of the dataframe using the provided list of one-hot encoded column names.
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
        column_name (str): The name of the column to be one-hot encoded.
        ohe_cols (list of str): A list of strings representing the names of the one-hot encoded columns.
    
    Returns:
        pd.DataFrame: A DataFrame containing the original data and the one-hot encoded columns.
    """
    # Extract the prefix to replace from the first entry assuming all entries are consistent
    prefix = ohe_cols[0].split('_')[0] + '_'
    
    # Create a set of unique categories expected from the ohe_cols
    expected_categories = {col.replace(prefix, '') for col in ohe_cols}
    
    # Initialize the DataFrame for the OHE columns with zeros
    ohe_df = pd.DataFrame(0, index=df.index, columns=ohe_cols)
    
    # Populate the DataFrame with 1s where appropriate
    for category in expected_categories:
        category_col_name = f"{prefix}{category}"
        # Check if the category is present in the current DataFrame
        ohe_df[category_col_name] = (df[column_name] == category).astype(int)
    
    # Concatenate the original DataFrame with the new OHE columns
    result_df = pd.concat([df, ohe_df], axis=1)
    
    return result_df

#extracts numbers from strings in pics2
def extract_number(s):
    # Ensure the input is a string
    s = str(s)
    
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    else:
        return 0


def generatePreds(data, threshold = 0.5):
    #Adds remaining features and generates predictions
    #data['pics2'] = data['pics2'].apply(extract_number)
    
    gmm3 = joblib.load('models/gmm_3_components.joblib')

    gmm_features = ['EMP','ESTAB','PAYANN', 'RCPTOT', 'RISK_SCORE','SOVI_SCORE','RESL_SCORE','24_hours','7_days', 'RATING', 'REVIEW_NUMBER', 'Census_Category']
    gmm_df = data.loc[:,gmm_features]

    #Need the below list for GMM cluster creation
    ohe_cols = ['Census_Category_Accommodation and food services', 'Census_Category_Accounting, tax preparation, bookkeeping, and payroll services', 'Census_Category_Administrative and support and waste management and remediation services', 'Census_Category_Administrative and support services', 'Census_Category_Administrative management and general management consulting services', 'Census_Category_Agencies, brokerages, and other insurance related activities', 'Census_Category_Ambulatory health care services', 'Census_Category_Amusement, gambling, and recreation industries', 'Census_Category_Architectural, engineering, and related services', 'Census_Category_Arts, entertainment, and recreation', 'Census_Category_Automobile dealers', 'Census_Category_Automotive body, paint, interior, and glass repair', 'Census_Category_Automotive mechanical and electrical repair and maintenance', 'Census_Category_Automotive parts and accessories stores', 'Census_Category_Automotive parts, accessories, and tire stores', 'Census_Category_Automotive repair and maintenance', 'Census_Category_Beauty salons', 'Census_Category_Building material and garden equipment and supplies dealers', 'Census_Category_Building material and supplies dealers', 'Census_Category_Clothing and clothing accessories stores', 'Census_Category_Clothing stores', 'Census_Category_Commercial banking', 'Census_Category_Computer systems design and related services', 'Census_Category_Educational services', 'Census_Category_Engineering services', 'Census_Category_Food and beverage stores', 'Census_Category_Food services and drinking places', 'Census_Category_Furniture and home furnishings stores', 'Census_Category_Gasoline stations', 'Census_Category_Gasoline stations with convenience stores', 'Census_Category_General freight trucking', 'Census_Category_General merchandise stores', 'Census_Category_General merchandise stores, including warehouse clubs and supercenters', 'Census_Category_Grocery stores', 'Census_Category_Hair, nail, and skin care services', 'Census_Category_Health care and social assistance', 'Census_Category_Individual and family services', 'Census_Category_Insurance carriers and related activities', 'Census_Category_Janitorial services', 'Census_Category_Landscaping services', 'Census_Category_Legal services', 'Census_Category_Management consulting services', 'Census_Category_Management, scientific, and technical consulting services', 'Census_Category_Miscellaneous store retailers', 'Census_Category_Nonstore retailers', 'Census_Category_Nursing and residential care facilities', 'Census_Category_Offices of chiropractors', 'Census_Category_Offices of dentists', 'Census_Category_Offices of lawyers', 'Census_Category_Offices of physicians', 'Census_Category_Other professional, scientific, and technical services', 'Census_Category_Other schools and instruction', 'Census_Category_Other services (except public administration)', 'Census_Category_Personal and laundry services', 'Census_Category_Pharmacies and drug stores', 'Census_Category_Professional, scientific, and technical services', 'Census_Category_Real estate', 'Census_Category_Real estate and rental and leasing', 'Census_Category_Religious, grantmaking, civic, professional, and similar organizations', 'Census_Category_Repair and maintenance', 'Census_Category_Restaurants and other eating places', 'Census_Category_Retail trade', 'Census_Category_Securities, commodity contracts, and other financial investments and related activities', 'Census_Category_Snack and nonalcoholic beverage bars', 'Census_Category_Sporting goods, hobby, musical instrument, and book stores', 'Census_Category_Telecommunications', 'Census_Category_Transportation and warehousing', 'Census_Category_Traveler accommodation', 'Census_Category_Truck transportation']

    #manually do OHE because of categories included in GMM model and not in demo data
    gmm_df = manual_one_hot_encode(gmm_df, 'Census_Category', ohe_cols)

    categorical_columns = ['Census_Category'] #should be approximately 100
    numerical_columns = [col for col in gmm_df.columns if col not in categorical_columns]
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_columns)
    ])
    gmm_df = preprocessor.fit_transform(gmm_df)

    probs = gmm3.predict_proba(gmm_df)
    prob_df = pd.DataFrame(probs, columns=['Gaussian_1', 'Gaussian_2', 'Gaussian_3'])
    #print("prob df:", prob_df.isnull().sum())

    #print("before merge: ", data.isnull().sum())
    Data_final = pd.concat([data.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)
    #print("after merge:", Data_final.isnull().sum())

    X_df = Data_final[['Day of Week', 'Day of Disaster', 'RISK_SCORE', 'SOVI_SCORE', 'RESL_SCORE', 'EMP', 'ESTAB', 'PAYANN', 'RATING', 'RCPTOT', 'REVIEW_NUMBER', 'pics2',
            'tempClosed_lag1', 'tempClosed_lag2', 'tempClosed_lag3', 'tempClosed_lag4', 'tempClosed_lag5',
            'distance_lag1', 'Gaussian_1', 'Gaussian_2', 'Gaussian_3']]
    
    X_df['pics2'] = X_df['pics2'].apply(extract_number)

    dow_ohe = ['Day of Week_Friday', 'Day of Week_Monday', 'Day of Week_Saturday', 'Day of Week_Sunday', 'Day of Week_Thrusday', 'Day of Week_Thursday', 'Day of Week_Tuesday', 'Day of Week_Wednesday']

    X_df = manual_one_hot_encode(X_df, 'Day of Week', dow_ohe)

    categorical_columns = ['Day of Week'] 
    numerical_columns = [col for col in X_df.columns if col not in categorical_columns]
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_columns)
    ])

    X = preprocessor.fit_transform(X_df)

     # Load Random Forest model
    rf_model = joblib.load('models/rf.joblib')
    
    # Predict probabilities of closure (encoded as 1)
    predictions = rf_model.predict_proba(X)[:, 1]  # Assuming that the second column is probability of class 1

    # Append predictions to the data DataFrame
    data['tempClosedProb'] = predictions

    # Optionally, use the threshold to make a closure decision
    data['ForecastedTempClosed'] = (data['tempClosedProb'] >= threshold).astype(int)

    return data.sort_values(by='tempClosedProb', ascending=False)

#gui_output = generatePreds(modelData)