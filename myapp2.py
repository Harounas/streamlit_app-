import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import dateutil.parser as parser
import altair as alt
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.dates import DateFormatter
import ssl
import requests
from io import StringIO
import seaborn as sns
ssl._create_default_https_context = ssl._create_stdlib_context
from datetime import datetime, timedelta

#from st_vizzu import *
from scipy.stats import chi2_contingency, fisher_exact
from scipy.stats import zscore


#url="https://docs.google.com/spreadsheets/d/1lyBADWC8fAhUNw4LOcIoOSYBqNeEbVs_KU71O8rKqfs/edit?usp=sharing"
url="https://docs.google.com/spreadsheets/d/1tDkXnZSyFfPo7z5pSEWDUxVSxHNk4iGLhNKXb60XPyU/edit?usp=sharing"
conn = st.connection("gsheets", type=GSheetsConnection)
st.title("Data summary sentinel project")
st.markdown("---")

conn = st.connection("gsheets", type=GSheetsConnection)

st.markdown("---")
@st.cache_data
def load_data(datapath):
    dataset = conn.read(spreadsheet=datapath)
    return dataset
uploaded_file =  st.sidebar.file_uploader("Upload a file", type=["txt", "csv", "xlsx"])
# Text input widget for token
token = st.sidebar.text_input("Input a token")
#token = st.sidebar.text_input("Input a token")


#df0 = load_data(url)
def main():
    
    # If a file is uploaded
    if uploaded_file is not None:
        #st.write("File uploaded successfully!")
        #st.write("File contents:")
        # Read and display file contents
        #df = uploaded_file.read()
        #st.write(df)
       
    
    # File uploader widget
    
      st.write("File uploaded:", uploaded_file.name)
     # Handle different file types
      if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        st.write("CSV file data:")
        #st.dataframe(df)
      elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
        st.write("Excel file data:")
        #st.dataframe(df)
      elif uploaded_file.name.endswith('.txt'):
        df = uploaded_file.read().decode("utf-8")
        st.write("Text file content:")
        #st.text(content)
    st.sidebar.title("Please  upload your own file  or Token")      
    
    if uploaded_file is None:
    # If a token is provided
     if token:
        #st.sidebar.header("Token provided:", token)
        data = {
    'token':token,
    'content': 'record',
    'action': 'export',
    'format': 'csv',
    'type': 'flat',
    'csvDelimiter': '',
    'fields[0]': 'participantid_site',
    'forms[0]': 'case_report_form',
   'forms[1]': 'sample_collection_form',
    'forms[2]': 'rdt_laboratory_report_form',
   'forms[3]': 'pcr_laboratory_report_form',
    'forms[4]': 'urinalysis_laboratory_report_form',
    'forms[5]': 'malaria_laboratory_report_form',
    'rawOrLabel': 'label',
    'rawOrLabelHeaders': 'raw',
    'exportCheckboxLabel': 'false',
    'exportSurveyFields': 'false',
    'exportDataAccessGroups': 'false',
    'returnFormat': 'csv'
}
        r = requests.post('https://redcap-acegid.org/api/',data=data)

        df = pd.read_csv(StringIO(r.text),  low_memory=False)
       # st.write(df.head())
     else:
        df=load_data(url)

    return df
  
  
df=main() 



   
#df = conn.read(spreadsheet=url)
df['date_crf'] = pd.to_datetime(df['date_crf'], errors='coerce', format='%Y-%m-%d')
df[['hiv_rdt','malaria_rdt','hepb_rdt','hepc_rdt','syphilis_rdt']]=df[['hiv_rdt','malaria_rdt','hepb_rdt','hepc_rdt','syphilis_rdt']].replace({1:'Positive',0:'Negative'})
df=df.dropna(subset=['siteregion_crf'])
df=df.dropna(subset=['date_crf'])
# Filter DataFrame based on current date and time
df = df[(df['date_crf']<=pd.to_datetime(datetime.now()))]
df=df.replace({'Ondo':'OWO','Lagos':'IKORODU','Ebonyi':'ABAKALIKI','Edo':'IRRUA'})
#df['Date of visit'] =pd.to_datetime(df['Date of visit'] ).dt.strftime('%Y-%m-%d')
col1, col2=st.sidebar.columns((2))
#Startdate=pd.to_datetime(df['date_crf']).min()
df['siteregion_crf']=df['siteregion_crf'].replace({1:'IKORODU',4:'ABAKALIKI',2:'OWO',3:'IRRUA'})
Enddate=pd.to_datetime(df['date_crf']).max()
Startdate=  pd.to_datetime(Enddate - timedelta(weeks=2))
with col1:
      date1=pd.to_datetime(st.date_input("Start Date", Startdate))
with col2:
      date2=pd.to_datetime(st.date_input("End Date", Enddate))

df=df[(df['date_crf']>=date1)&(df['date_crf']<=date2)].copy()

st.sidebar.header("Choose your filter : ")
##Create for State

state=st.sidebar.multiselect("Choose a State",   df["siteregion_crf"].unique())
if not state:
    df1=df.copy()
else:
   df1=df[df['siteregion_crf'].isin(state)]
 
 
dfsample=df[['siteregion_crf','bloodsample','urinesample',
'nasosample',
'salivasample',
'oralsample']]
samplecol=['siteregion_crf','bloodsample','urinesample',
'nasosample',
'salivasample',
'oralsample']
testcol=['siteregion_crf','hiv_rdt','malaria_rdt','hepb_rdt','hepc_rdt','syphilis_rdt']
dftest=df[['siteregion_crf','hiv_rdt','malaria_rdt','hepb_rdt','hepc_rdt','syphilis_rdt']]
dfpcr=df[['siteregion_crf','yellowfever_pcr','lassa_pcr','ebola_pcr','marburg_pcr','westnile_pcr','zika_pcr','cchf_pcr','riftvalley_pcr','dengue_pcr','ony_pcr','covid_pcr','mpox_pcr']]
pcrcol=['siteregion_crf','yellowfever_pcr','lassa_pcr','ebola_pcr','marburg_pcr','westnile_pcr','zika_pcr','cchf_pcr','riftvalley_pcr','dengue_pcr','ony_pcr','covid_pcr','mpox_pcr']
def weeksago(df):
  two_weeks_ago =  pd.to_datetime(Startdate - timedelta(weeks=2))
  sliced_df = sliced_df =df[(df['date_crf']<=date1)&(df['date_crf']>=two_weeks_ago)].copy()
  
  return sliced_df 
def samplecollected(df):
     dfsample1=dfsample.drop(columns=['siteregion_crf'])
     missingfomrs = dfsample1[dfsample1.isna().all(axis=1)]
     return missingfomrs
     
def sampletest(df):
     dftest1=dftest.drop(columns=['siteregion_crf'])
     missingfomrs = dftest1[dftest1.isna().all(axis=1)]
     return missingfomrs
def samplepcr(df):
     dfpcr1=dfpcr.drop(columns=['siteregion_crf'])
     missingfomrs = dfpcr1[dfpcr1.isna().all(axis=1)]
     return missingfomrs          

# Calculate the date two weeks ago
two_weeks_ago = pd.to_datetime(date1 - timedelta(weeks=2))
dfsample1=samplecollected(df)
dftest1=sampletest(df)
dfpcr1=samplepcr(df)

sliced_df =df[(df['date_crf']>=date1)&(df['date_crf']<=two_weeks_ago)].copy()
data1 = {' ':['Patient enrolled','Sample collected','RDTs run','PCRs run'],
        'Last two weeks': [len(df),"{} ({:.2f}%)".format(len(df)*5,(((len(df)*5-dfsample1.isnull().sum().sum())/(len(df)*5))*100)),"{} ({:.2f}%)".format(len(df)*5,(((len(df)*5-dftest1.isnull().sum().sum())/(len(df)*5))*100)),"{} ({:.2f}%)".format(len(df)*12,(((len(df)*12-dfpcr1.isnull().sum().sum())/(len(df)*12))*100))]
    
      #  'previous two weeks': [len(weeksago(df)),"{} ({:.2f}%)".format(len(weeksago(df))*5,(((len(weeksago(df))-len(samplecollected(weeksago(df))))/len(weeksago(df)))*100))]
        }
#"{} {.2f%} ".format(len(df),((len(dfsample)*5-samplecollected(df)*5/len(df)*5)*100)),((len(weeksago(df))*5-samplecollected(weeksago(dfsample))*5/len(weeksago(dfsample))*5)*100)

    # Create DataFrame
data11 = {' ':['Patient enrolled','Sample collected','RDTs run','PCRs run'],
        'Last two weeks': [len(df),"{} ({:.2f}%)".format(len(df)*5,((len(df)-len(dfsample1))/len(df))*100),"{} ({:.2f}%)".format(len(df)*5,((len(df)-len(dftest1))/(len(df)))*100),"{} ({:.2f}%)".format(len(df)*12,((len(dfpcr1)-len(dfpcr1))/len(df))*100)]
    
      #  'previous two weeks': [len(weeksago(df)),"{} ({:.2f}%)".format(len(weeksago(df))*5,(((len(weeksago(df))-len(samplecollected(weeksago(df))))/len(weeksago(df)))*100))]
        }
dataframe1 = pd.DataFrame(data1)

    # Display the DataFrame as a table
#st.dataframe(dataframe1) 

data2 = {' ':['HIV','Malaria','Hepatitis B','Hepatitiis C','Syphilis'],
        'Last two weeks': ["{} ({:.2f}%)".format((df['hiv_rdt'] == 'Positive').sum(),(df['hiv_rdt'] == 'Positive').sum()*100/len(df)),"{} ({:.2f}%)".format((df['malaria_rdt'] == 'Positive').sum(),(df['malaria_rdt'] == 'Positive').sum()*100/len(df)),"{} ({:.2f}%)".format((df['hepb_rdt'] == 'Positive').sum(),(df['hepb_rdt'] == 'Positive').sum()*100/len(df)),"{} ({:.2f}%)".format((df['hepc_rdt'] == 'Positive').sum(),(df['hepc_rdt'] == 'Positive').sum()*100/len(df)),"{} ({:.2f}%)".format((df['syphilis_rdt'] == 'Positive').sum(),(df['syphilis_rdt'] == 'Positive').sum()*100/len(df))]
    
      #  'previous two weeks': [len(weeksago(df)),"{} ({:.2f}%)".format(len(weeksago(df))*5,(((len(weeksago(df))-len(samplecollected(weeksago(df))))/len(weeksago(df)))*100))]
        }
#"{} {.2f%} ".format(len(df),((len(dfsample)*5-samplecollected(df)*5/len(df)*5)*100)),((len(weeksago(df))*5-samplecollected(weeksago(dfsample))*5/len(weeksago(dfsample))*5)*100)

    # Create DataFrame
dataframe2 = pd.DataFrame(data2)

    # Display the DataFrame as a table
#st.dataframe(dataframe2)
st.write(df.head())    
st.write("## Overall summary:summary")
col1, col2=st.columns((2))

with col1:
    st.dataframe(dataframe1) 
 
with col2:
    st.dataframe(dataframe2) 
  
for state in df['siteregion_crf'].unique():
   dfs=df[df['siteregion_crf']==state]  
   #dfsample=dfsample[dfsample['siteregion_crf']==state]
   #dftest=dftest[dftest['siteregion_crf']==state]
   #st.write(dfsample)
   dfsample=dfs[samplecol]
   dfsample=dfs[samplecol]
   dftest=dfs[testcol]
   dfpcr=dfs[pcrcol]
   dftest=dftest.drop(columns=['siteregion_crf'])
   missingformt = dftest[dftest.isna().all(axis=1)]
   dfpcr=dfpcr.drop(columns=['siteregion_crf'])
   missingformp = dfpcr[dfpcr.isna().all(axis=1)]
   dfsample=dfsample.drop(columns=['siteregion_crf'])
   missingforms = dfsample[dfsample.isna().all(axis=1)]

   data = {' ':['Patient enrolled','Missing sample collection forms',' Missing RDTs forms','Missing PCRs forms'],
        'Last two weeks': [len(dfs),"{} ({:.2f}%)".format( missingforms.isna().sum().sum(),(( missingforms.isna().sum().sum()/(len(dfs)*5))*100)),"{} ({:.2f}%)".format( missingformt.isna().sum().sum(),(( missingformt.isna().sum().sum()/(len(dfs)*5))*100)),"{} ({:.2f}%)".format( missingformp.isna().sum().sum(),(( missingformp.isna().sum().sum()/(len(dfs)*12))*100))]}
   dataframe = pd.DataFrame(data) 
   st.write(f"## {state}")
   st.dataframe(dataframe)    

dff= df.groupby(['date_crf', 'siteregion_crf']).size().reset_index(name='Count')
#dff['Region of site']=dff['Region of site'].replace({'Ikorodu':'IKORODU','Abakaliki ':'Abakaliki','Abakaliki ':'Abakalik','Ebonyi ':'Ebonyi'})

#dff= dff[(dff['LGA']=="IKORODU") | (dff['LGA'] == 'OWO') |(dff['LGA'] == 'Abakaliki')|(dff['LGA'] == 'Ebonyi')]
#dff= dff[(dff['Region of site']=='IKORODU') | (dff['Region of site'] == 'Abakaliki')]
dff['date_crf'] =pd.to_datetime(dff['date_crf'] ).dt.strftime('%Y-%m-%d')
dfff=pd.merge(dff,dff.groupby(['date_crf']).sum().reset_index(),on="date_crf")
dfff['Total']='Total'
fig,ax = plt.subplots(figsize=(15, 12))
sns.lineplot( x="date_crf", y="Count_x", data=dfff , hue='siteregion_crf',palette='Set1').set(title=' ', xlabel='Date', ylabel='siteregion_crf')
sns.lineplot( x="date_crf", y="Count_y", data=dfff,hue='Total',palette=['black'],).set(title=' ', xlabel='Date', ylabel='siteregion_crf')
#sns.set_theme(style='white', font_scale=3)
ax.legend(loc='upper center', #bbox_to_anchor=(0.4,0.0001),
          fancybox=True, shadow=True, ncol=5)
# Remove the frame (border) around the plot
#sns.gca().spines['top'].set_visible(False)
#sns.gca().spines['right'].set_visible(False)
#plt.gca().spines['bottom'].set_visible(False)
#plt.gca().spines['left'].set_visible(False)
#monthly_ticks = pd.date_range(start=dff['Date of visit (dd/mm/yyyy)'].iloc[0], end=dff['Date of visit (dd/mm/yyyy)'].iloc[-1],freq='d')  # Monthly intervals
#plt.xticks(ticks=monthly_ticks, labels=[date.strftime('%Y-%m-%d') for date in monthly_ticks], rotation=45)

ax.tick_params(axis='x', labelsize=10)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
#ax.set_title('Plot through Time with Custom X-axis Ticks')
plt.xticks(rotation=45)
#ax.tight_layout()
st.pyplot(fig)

dff= df.groupby(['hiv_rdt', 'siteregion_crf']).size().reset_index(name='Count')

dfff=pd.merge(dff,dff.groupby(['siteregion_crf']).sum().reset_index(),on="siteregion_crf")


dfc=df[(df['participantid_crf']!=np.nan)&(df['participantid_rdt']!=np.nan)&(df['participantid_rdt']!=df['participantid_crf'])&(df['age_rdt']!=df['age_crf'])|(df['sex_rdt']!=df['sex_crf'])]
dfc=dfc.dropna(subset=['participantid_crf','participantid_rdt'])
st.write(f"## Mismatch check between forms: number of mismach is {len(dfc)} ")
st.write(dfc[['participantid_crf','participantid_rdt','age_rdt','age_crf','sex_crf','sex_rdt' ,'siteregion_crf']])     
        
select_out=st.multiselect('Please select a to check outliers:',df.select_dtypes(include='number').columns)
# Function to detect outliers using z-score
def detect_outliers_zscore(data):
    # Calculate z-score for each value in the dataframe
    z_scores = np.abs(zscore(data))
    # Threshold for considering a value as an outlier (e.g., z-score > 3)
    threshold = 3
    # Create a boolean mask indicating outliers
    outlier_mask = (z_scores > threshold)
    return outlier_mask

# Main app
if select_out:
    data =df[select_out]



    # Checkbox to show outliers
    show_outliers = st.checkbox('Show Outliers')

    # Detect outliers using z-score
    outliers = detect_outliers_zscore(data)
    st.write(outliers)
    if len(outliers)!=0:
        # Show outliers in the dataframe
        st.write('Outliers:', data[outliers.any(axis=1)])
    else:
        st.write('No outliers detected.')
        
           
        
select_catcol=st.multiselect('Please select categorical column to make  a bar plot:',df.select_dtypes(include='object').columns)

if select_catcol:
    st.write("Selected categorical column is : ", select_catcol[0])
 
   # selected_columns = st.multiselect("select column", select_col)
    s = df[select_catcol[0]].str.strip().value_counts()
    count_df = pd.DataFrame({f'{select_catcol[0]}': s.index, 'Count': s.values})
    st.write(count_df)
    trace = go.Bar(x=s.index,y=s.values,showlegend = False, text=s.values)
    layout = go.Layout(title = f"Bar plot for {select_catcol[0]}")
    data = [trace]
    fig = go.Figure(data=data,layout=layout)
    st.plotly_chart(fig)
else:
    st.info('Please select a categorical column using the dropdown above.')


select_numcol=st.multiselect('Please select numerical column to make  a histogram plot:',df.select_dtypes(include='number').columns)




#select_numcol=st.multiselect('Select numerical column:',dfs.select_dtypes(include='number').columns)
#select_numcol=st.multiselect('Select numerical column:',dfs.columns)

if select_numcol:
   st.write("Selected numerical column is : ", select_numcol[0])
   fig, ax = plt.subplots()
   q1 = df[select_numcol[0]].quantile(0.25)
   q3 = df[select_numcol[0]].quantile(0.75)
   iqr = q3 - q1
   bin_width = (2 * iqr) / (len(dfs[select_numcol[0]]) ** (1 / 3))
   bin_count = int(np.ceil((dfs[select_numcol[0]].max() - dfs[select_numcol[0]].min()) / bin_width))
   selected_value = st.slider('Select number of bins:', min_value=0, max_value=100, value=bin_count)

# Display the selected value
   st.write(f'Select number of bins: {selected_value}')
   ax.hist(dfs[select_numcol[0]], bins=selected_value, edgecolor='black')
   st.pyplot(fig)
 
else:
    st.info('Please select a numerical column using the dropdown above.')


select_catcol=st.multiselect('Please select categorical column to make  a box plot:',df.select_dtypes(include='object').columns)
select_numcol=st.multiselect('Please select categorical column to make  a box plot:',df.select_dtypes(include='number').columns)

def t_test(sample1, sample2):
    t_statistic, p_value = stats.ttest_ind(sample1, sample2)
    return t_statistic, p_value

def mann_whitney_test(sample1, sample2):
    u_statistic, p_value = stats.mannwhitneyu(sample1, sample2)
    return u_statistic, p_value

if select_catcol and select_numcol:
   cat=st.multiselect("Choose two categories",   df[select_catcol[0]].unique())
   if not cat:
       df1=df.copy()
       df1[select_numcol[0]]=df1[select_numcol[0]].fillna(df1[select_numcol[0]].mean())
       sample1=df1.loc[df1[select_catcol[0]]==df1[select_catcol[0]].value_counts().index.tolist()[0]][select_numcol[0]]
       sample2=df1.loc[df1[select_catcol[0]]==df1[select_catcol[0]].value_counts().index.tolist()[1]][select_numcol[0]]
   else:
      df1=df[df[select_catcol[0]].isin(cat)] 
      df1[select_numcol[0]]=df1[select_numcol[0]].fillna(df1[select_numcol[0]].mean())
      sample1=df1.loc[df1[select_catcol[0]]==cat[0]][select_numcol[0]]
      sample2=df1.loc[df1[select_catcol[0]]==cat[1]][select_numcol[0]]
   
   st.write("Selected categorical column is : ", select_catcol[0], "" " and Selected numerical column is : ",select_numcol[0] )
   fig, ax = plt.subplots()
   df1.boxplot(by=select_catcol[0], column=select_numcol[0], ax=ax)
   plt.title(' ')
   plt.xlabel(f'{select_catcol[0]}')
   plt.ylabel(f'{select_numcol[0]}')

# Display the plot in Streamlit app
   st.pyplot(fig)
   #st.write(dfs.loc[dfs[select_catcol[0]]==dfs[select_catcol[0]].value_counts().index.tolist()[0]]['Sex'])
   
   # Perform Mann-Whitney U test
   
   #sample1=df1.loc[df1[select_catcol[0]]==df1[select_catcol[0]].value_counts().index.tolist()[0]][select_numcol[0]]
  # sample2=df1.loc[df1[select_catcol[0]]==df1[select_catcol[0]].value_counts().index.tolist()[1]][select_numcol[0]]
   # Perform t-test
   t_statistic, t_p_value = t_test(sample1, sample2)

    # Perform Mann-Whitney U test
   u_statistic, u_p_value = mann_whitney_test(sample1, sample2)

    # Display results
   st.write("### Statistical test")
   st.write("#### Independent Samples t-test")
   st.write(f"T-Statistic: {t_statistic}")
   st.write(f"P-Value: {t_p_value}")

   st.write("#### Mann-Whitney U Test")
   st.write(f"U-Statistic: {u_statistic}")
   st.write(f"P-Value: {u_p_value}")


# Function to perform chi-square test
def chi_square_test(data1,data2):
    data=pd.crosstab(data1, data2)
    chi2, p, dof, expected = chi2_contingency(data)
    return chi2, p

# Function to perform Fisher's exact test
def fishers_exact_test(data):
    oddsratio, p = fisher_exact(data)
    return oddsratio, p
select_col11=st.multiselect('Please select a first categorical column for statisticala analysis',df.select_dtypes(include='object').columns)
select_col22=st.multiselect('Please select a second categorical column for statisticala analysis',df.select_dtypes(include='object').columns)
# Main app
if select_col11 and select_col22:
   cat11=st.multiselect("Choose two categories for the first categorical variable",   df[select_col11[0]].unique())
   cat22=st.multiselect("Choose two categories for the first categorical variable",   df[select_col22[0]].unique())
   if not cat11 and not cat22:
       df1=df.copy()
 
   elif cat11 and not cat22:
     df1=df[df[select_col11[0]].isin(cat11)] 
  
   else:
     df1=df[df[select_col22[0]].isin(cat22)]  
   
    # Filter data to selected columns
 
   selected_data = df1[[select_col11[0], select_col22[0]]].dropna()
   
    # Perform chi-square test
   st.subheader('## Chi-Square Test')
   chi2, p = chi_square_test(selected_data[select_col11[0]], selected_data[select_col22[0]])
   st.write('Chi-Square Statistic:', chi2)
   st.write('P-value:', p)
   
    # Perform Fisher's exact test
   st.subheader("## Fisher's Exact Test")
   contingency_table = pd.crosstab(selected_data[select_col11[0]], selected_data[select_col22[0]])
   oddsratio, p = fishers_exact_test(contingency_table)
   st.write('Odds Ratio:', oddsratio)
   st.write('P-value:', p)












