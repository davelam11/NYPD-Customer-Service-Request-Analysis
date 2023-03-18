import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import math

pd.options.display.width=None
pd.options.display.max_columns=None
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)

"""
Step 1.1 - Identify the Shape of the dataset
"""
print("========== Step 1.1 - Identify the Shape of the dataset ==========")
df = pd.read_csv("311_Service_Requests.csv",low_memory=False)
print(df.shape)

"""
Step 1.2 - Identify variables with null values
"""
print("========== Step 1.2 - Identify variables with null values ==========")
var_null_list = []
for column in df.columns:
    a = df[column].isnull().values.any()
    b = df[column].isnull().sum()
    if b > 0:
        var_null_list.append(str(column))
        print(str(column) + ": This variable have null values of " + str(b) + " in total.")
    else:
        print(str(column) + ": This variable don't have any null value.")

leng = len(var_null_list)
print("There is in total " + str(leng) + " variables with null values.")
print(var_null_list)

"""
Step 2.1 - Missing value treatment
"""
print("========== Step 2.1 - Missing value treatment ==========")
#--Printing the dtype and number of Null values of each variable
for column in df.columns:
    print(column, "//", df[column].dtype, "//", df[column].isna().sum())

#--Dropping insignificant columns and rows with NaN values
df = df.drop(columns=['Agency Name','Landmark','School or Citywide Complaint','Vehicle Type','Taxi Company Borough',\
                      'Taxi Pick Up Location','Bridge Highway Name','Bridge Highway Direction','Road Ramp',\
                      'Bridge Highway Segment','Garage Lot Name','Ferry Direction','Ferry Terminal Name','Latitude',\
                      'Longitude','Location'])

df = df.drop(columns=['School Name','School Number','School Region','School Code','School Phone Number','School Zip',\
                      'School Address','School City','School State','School Not Found'])

df = df.dropna(subset=["City","Incident Zip"])

#--Fillna to Selected String Variables using values dictionary
values = {"Descriptor": "None", "Location Type": "Street/Sidewalk", "Incident Address": "NaN", "Street Name": "NaN",\
          "Cross Street 1": "NaN", "Cross Street 2": "NaN", "Intersection Street 1": "NaN",\
          "Intersection Street 2": "NaN", "Address Type": "ADDRESS", "Facility Type": "Precinct"}

df = df.fillna(value=values)

#--Fillna to Datetime variables
out_format = "%m/%d/%Y %H:%M:%S"
input_format = "%m/%d/%Y %I:%M:%S %p"
created_date_list = []
closed_date_list = []
createdate_check_arr = pd.Index(df["Created Date"]).notnull()
closedate_check_arr = pd.Index(df["Closed Date"]).notnull()

for i, q, r, s in zip(df["Created Date"], list(createdate_check_arr), df["Closed Date"], list(closedate_check_arr)):
    if (q == True) and (s == True):
        createdt_obj = datetime.strptime(i,input_format)
        createdt_str = createdt_obj.strftime(out_format)
        dt_object_1 = datetime.strptime(createdt_str,out_format)
        created_date_list.append(dt_object_1)

        closedt_obj = datetime.strptime(r, input_format)
        closedt_str = closedt_obj.strftime(out_format)
        dt_object_2 = datetime.strptime(closedt_str, out_format)
        closed_date_list.append(dt_object_2)
    else:
        pass

df = df[df["Closed Date"].notnull() == True]  #starting from here only data with valid Close Date will be processed.
df["Created Date 24H"] = created_date_list
df["Closed Date 24H"] = closed_date_list
df["Due Date 24H"] = df["Created Date 24H"] + timedelta(hours=8)

res_date_list = []
resdate_check_arr = pd.Index(df["Resolution Action Updated Date"]).notnull()
for i, q in zip(df["Resolution Action Updated Date"], list(resdate_check_arr)):
    if q == True:
        resdt_obj = datetime.strptime(i, input_format)
        resdt_str = resdt_obj.strftime(out_format)
        dt_object_3 = datetime.strptime(resdt_str, out_format)
        res_date_list.append(dt_object_3)
    else:
        res_date_list.append(None)

df["Resolution Action Updated Date 24H"] = res_date_list
df["Resolution Action Updated Date 24H"] = df["Resolution Action Updated Date 24H"].fillna(df["Closed Date 24H"])

#--Dropping unused raw date columns
df = df.drop(columns=["Created Date","Closed Date","Due Date","Resolution Action Updated Date"])

"""
Step 2.2 - Remove Incorrect Timeline Rows, and assigning a new row of response time in seconds to the df
"""
print("===== Step 2.2 - Remove Incorrect Timeline Rows and Assigning Response Time Columns to the dataframe =====")
response_time_list = [closed - created for closed, created in zip(closed_date_list, created_date_list)]
response_time_seconds_list = [i.total_seconds() for i in response_time_list]
response_mean_seconds = math.ceil((sum(response_time_seconds_list)/len(response_time_list)))

def convert(n):
    return str(timedelta(seconds = n))


df["Response Time Seconds"] = response_time_seconds_list

for i in df["Response Time Seconds"]:
    if i < 0:
        df = df.drop(i)
    else:
        pass

"""
Step 2.3 - Draw a Frequency Plot for City-Wise Complaints
"""
print("========== Step 2.3 - Draw a Frequency Plot for City-Wise Complaints ==========")
city_list = df["City"].unique()
city_counter = df["City"].value_counts(sort=False)

#--Create a separate dataframe and plotting the bar chart for city-wise complaints counts
newdf = pd.DataFrame({'Case Count':list(city_counter)},index=city_list)
newdf.plot(kind="bar")
plt.title("City-wise Complaints Count Chart")
plt.xlabel("City Name")
plt.ylabel("No. of Case")
plt.show()

"""
Step 2.4 - Scatter and Hexbin Plots for complaint concentration across Brooklyn
"""
print("===== Step 2.4 - Scatter and Hexbin Plots for complaint concentration across Brooklyn =====")
brooklyn_df = df[df["Borough"] == "BROOKLYN"]
x = brooklyn_df["X Coordinate (State Plane)"]
y = brooklyn_df["Y Coordinate (State Plane)"]

xtick_arr = np.linspace(min(brooklyn_df["X Coordinate (State Plane)"]),max(brooklyn_df["X Coordinate (State Plane)"]),20,endpoint=True)
ytick_arr = np.linspace(min(brooklyn_df["Y Coordinate (State Plane)"]),max(brooklyn_df["Y Coordinate (State Plane)"]),20,endpoint=True)

#Show the Scatter Diagram
plt.scatter(x, y, s=7, alpha=0.5)
plt.xticks(xtick_arr, rotation=90)
plt.yticks(ytick_arr)
plt.title("Brooklyn Complaints Concentration Scatter Diagram")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.show()

#Show the Hexbin Plot
plt.hexbin(x, y, gridsize=(50,50))
plt.title("Brooklyn Complaints Concentration Hexbin Plot")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.show()

"""
Step 3.1 - Bar Graph of Count vs Complaint Types
"""
print("========== Step 3.1 - Bar Graph of Count vs Complaint Types ==========")
complaint_type_list = df["Complaint Type"].unique()
case_counter = df["Complaint Type"].value_counts(sort=False)

bardf = pd.DataFrame({'Case Count':list(case_counter)},index=complaint_type_list)
bardf.plot(kind="bar")
plt.title("Complaint Types Bar Chart")
plt.xlabel("Complaint Types")
plt.ylabel("No. of Case")
plt.show()

"""
Step 3.2 - Top 10 Types of Complaints
"""
print("========== Step 3.2 - Top 10 Types of Complaints ==========")
top10_counter = df["Complaint Type"].value_counts(sort=True).head(10)
print(top10_counter)

"""
Step 3.3 - Types of Complaints in each city
"""
print("========== Step 3.3 - Types of Complaints in each city ==========")
city_dic = {}
for city in city_list:
    city_df = df[df["City"] == city]
    city_dic[city] = city_df["Complaint Type"].value_counts(sort=True)

print(city_dic)

"""
Step 4 - Major Types of Complaints in Each City
"""
print("========== Step 4 - Major Types of Complaints in Each City ==========")
for key, i in zip(city_dic, range(len(city_list))):
    city_dic[key].plot(kind="bar")
    plt.title("Major types of complaints in {}".format(city_list[i]))
    plt.xlabel("Complaint Types")
    plt.ylabel("No. of Case")
    plt.show()

"""
Step 5 - Average Response Time across Various Types of Complaints
"""
print("========== Step 5 - Average Response Time across Various Types of Complaints ==========")
for type in complaint_type_list:
    complaint_type_df = df[df["Complaint Type"] == type]
    converted_datetime = convert(math.ceil(complaint_type_df["Response Time Seconds"].mean())) #using the self-defined function convert(n)
    print(type, ":", converted_datetime)

"""
Step 6 - Statistical Analysis of Significant Variables

Upon exploration of the dataset I think that there could be statistical relationship between either
"Borough"/ "City"/ "Complaint Type"/ "Location Type" and the "Resolution Description" variable.
As "Resolution Description" describes how the complaints will be eventually handled, this variable is
obviously the "result" of the whole dataset. And "Borough"/ "City"/ "Complaint Type"/ "Location Type"
could be possible factors affecting what the final resolution action for each complaint should be.

Thus I am going to prove the relationship between these possible factors and the resolution action by
carrying out the Chi-square test using chi2_contingency function from Scipy.

"""
print("========== Step 6 - Statistical Analysis of Significant Variables ==========")
#-----Encoding Chosen Categorical Variables-----#
selected_column = ["Borough","City","Complaint Type","Location Type","Resolution Description"]

for column in selected_column:
    df[column] = df[column].astype("category")
    df[column+"_Cat"] = df[column].cat.codes

#-----Pairing of factor variables to target variables into a list of list-----#
var_pair_list = [["Borough", "Resolution Description_Cat"],["City", "Resolution Description_Cat"],\
             ["Complaint Type", "Resolution Description_Cat"],["Location Type", "Resolution Description_Cat"]]

#-----Print out all the unique type of resolution for easier comparison later-----#
resolution_unique_list = df["Resolution Description"].value_counts().index.tolist()
print("Type of Resolution are as below:")
for i, q in zip(resolution_unique_list, range(len(resolution_unique_list))):
    print(q,":", i)

#-----Generate the contingency table and carrying out chi-squared test for each variable combination-----#
for pair in var_pair_list:
    contingent_df = pd.DataFrame({pair[0]: df[pair[0]], pair[1]: df[pair[1]]})
    contingent_table = pd.crosstab(index=contingent_df[pair[0]], columns=contingent_df[pair[1]], margins=True)
    chi2, p, dof, ex = chi2_contingency(contingent_table, correction=False)
    print("=================================================================================================")
    print("           Statistical Analysis of ", pair[0], "~", pair[1], " combination")
    print("=================================================================================================")
    print(contingent_table)
    print("-------------------------------------------------------------------------------------------------")
    print("Chi-Square Statistic: ", chi2)
    print("Original p-Value: ", p)
    print("10-digits-rounded p-Value: ", '{:.10f}'.format(p))
    print("Degree of Freedom: ", dof)
    if p < 0.05:
        print("Conclusion: The null hypothesis is successfully rejected and there is significant statistical relationship between the two variables.")
    else:
        print("Conclusion: Fail to reject the null hypothesis and no relationship between the two variables is found.")
    print("-------------------------------------------------------------------------------------------------")
