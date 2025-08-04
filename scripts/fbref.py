import requests
import random
import time
import re
import ast
import numpy as np
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import date


### FBREF SECTION

headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '3600',
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
    }

def get_proxy():
    r = requests.get('https://www.us-proxy.org/')
    soup = BeautifulSoup(r.content, 'html.parser')
    table = soup.find_all('table')[0]
    df = pd.read_html(StringIO(str(table)))[0]  # Use StringIO to wrap HTML
    i = random.randint(0, df.index.size - 3)
    proxy_string = "{'http': '" + df.loc[i]['IP Address'] + ':' + df.loc[i]['Port'].astype('str') + "'}"
    proxy = ast.literal_eval(proxy_string)
    return proxy

def extract_date_and_league(url):
    # Regular expression to match the date and league part
    match = re.search(r'(\w+-\d+-\d+)-(.*)', url)
    if match:
        date_str = match.group(1)
        league_str = match.group(2)
        return pd.to_datetime(date_str), league_str
    return None, None


def clean_team_name_fbref(team_name):
    # Check if team_name is a string before applying regex
    if isinstance(team_name, str):
        # Define a list of substrings to remove as whole words
        remove_list = ['dk', 'it', 'at', 'eng', 'es', 'de', 'rs', 'fr', 'sct', 'nl', 'pt', 'ua', 'az', 'si', 'tr', 'ad', 'gr', 
                       'hu', 'ge', 'cy', 'ch', 'cz', 'xk', 'md', 'be', 'by', 'kz', 'is', 'pl', 'wls', 'se', 'il', 'sk', 'hr', 
                       'bg', 'ba', 'no', 'gi']

        # Construct a regex pattern that matches any of the substrings surrounded by word boundaries
        pattern = r'\b(?:' + '|'.join(re.escape(substring) for substring in remove_list) + r')\b'

        # Replace the matched substrings with an empty string
        team_name = re.sub(pattern, '', team_name)

        # Remove any extra spaces left after substitution and return the cleaned name
        return re.sub(r'\s+', ' ', team_name).strip()
    else:
        # If the team_name is not a string, return it as is
        return team_name



def refresh_fbref_data(df):
  df['start_time_pt'] = pd.to_datetime(df['start_time']).dt.tz_convert('US/Pacific')
  date_list=df.sort_values('start_time_pt',ascending=True)['start_time_pt'].dt.date.astype(str).unique().tolist()
  ## last 5 elements from date_list
  date_list_recent = date_list[-25:]

  df_all = pd.read_parquet('./data/fb_ref_data.parquet', engine='pyarrow')

  for date in date_list_recent:
    print(date)
    proxy=get_proxy()
    sleep_time=2.0 + np.random.uniform(1,4) +  np.random.uniform(0,1)
    time.sleep(sleep_time)


    url=f'https://fbref.com/en/matches/{date}'
    r = requests.get(url,headers=headers,proxies=proxy)


    soup = BeautifulSoup(r.content, "html.parser")
    all_urls = []
    for td_tag in soup.find_all('td', {"class":"center"}):
        if 'href' in str(td_tag):
            all_urls.append(
                "https://fbref.com" +str(td_tag).split('href="')[1].split('">')[0]
            )


    dfs = pd.read_html(url, header=0, index_col=0)
    df = pd.DataFrame(dfs[0])
    for i in range(1, len(dfs)):
        df = pd.concat([df,dfs[i]])
    df=df.query('Score.notnull()')
    df = df.loc[df.Home!='Home']
    df.reset_index(drop=False,inplace=True)
    df['url'] = pd.Series(all_urls)
    df['match_selector'] = df['Home']+' '+df['Score']+' '+df['Away']
    df['date_scraped'] = datetime.now()
    df['date'] = date
    df=df.rename(columns={'xG':'home_xg', 'xG.1':'away_xg'})


    df_all = pd.concat([df_all,df])
    print(df_all.index.size)
    time.sleep(4)


  df_all = df_all.drop_duplicates(subset=['date','Home','Away','Venue','Score'], keep='last')
  # df_all.to_csv('/content/drive/MyDrive/Analytics/fbref_match_data.csv',index=False)

  df_all['Home'] = df_all['Home'].apply(clean_team_name_fbref)
  df_all['Away'] = df_all['Away'].apply(clean_team_name_fbref)
  df_all['Home'] = df_all['Home'].apply(clean_team_name_fbref)
  df_all['Away'] = df_all['Away'].apply(clean_team_name_fbref)
  df_all['date_scraped'] = df_all['date_scraped'].astype(str)

  try:
    df_all['Wk'] = pd.to_numeric(df_all['Wk'], errors='coerce')
    df_all['Wk'].fillna(0, inplace=True)  # Or drop the rows: df.dropna(subset=['Wk'], inplace=True)
  except Exception as e:
    print(f'fb ref failure {e}')

  
  # Attempt to convert the 'Attendance' column to numeric, coercing errors
  df_all['Attendance'] = pd.to_numeric(df_all['Attendance'], errors='coerce')
  
  # Now, you can proceed with creating a parquet table
  table = pa.Table.from_pandas(df_all)
  pq.write_table(table, './data/fb_ref_data.parquet', compression='BROTLI')
  print('fb ref data written')




def refresh_fbref_data_updated(df):
  max_date = df.date.max()
  current_date = date.today()
  dates = pd.date_range(start=max_date, end=current_date).strftime('%Y-%m-%d').tolist()

  df_all = pd.read_parquet('./data/fb_ref_data.parquet', engine='pyarrow')

  for dt in dates:
    print(dt)
    proxy=get_proxy()
    sleep_time=2.0 + np.random.uniform(1,4) +  np.random.uniform(0,1)
    time.sleep(sleep_time)


    url=f'https://fbref.com/en/matches/{dt}'
    r = requests.get(url,headers=headers,proxies=proxy)


    soup = BeautifulSoup(r.content, "html.parser")
    all_urls = []
    for td_tag in soup.find_all('td', {"class":"center"}):
        if 'href' in str(td_tag):
            all_urls.append(
                "https://fbref.com" +str(td_tag).split('href="')[1].split('">')[0]
            )


    dfs = pd.read_html(url, header=0, index_col=0)
    df = pd.DataFrame(dfs[0])
    for i in range(1, len(dfs)):
        df = pd.concat([df,dfs[i]])
    df=df.query('Score.notnull()')
    df = df.loc[df.Home!='Home']
    df.reset_index(drop=False,inplace=True)
    df['url'] = pd.Series(all_urls)
    df['match_selector'] = df['Home']+' '+df['Score']+' '+df['Away']
    df['date_scraped'] = datetime.now()
    df['date'] = dt
    df=df.rename(columns={'xG':'home_xg', 'xG.1':'away_xg'})


    df_all = pd.concat([df_all,df])
    print(df_all.index.size)
    time.sleep(4)


  df_all = df_all.drop_duplicates(subset=['date','Home','Away','Venue','Score'], keep='last')
  # df_all.to_csv('/content/drive/MyDrive/Analytics/fbref_match_data.csv',index=False)

  df_all['Home'] = df_all['Home'].apply(clean_team_name_fbref)
  df_all['Away'] = df_all['Away'].apply(clean_team_name_fbref)
  df_all['Home'] = df_all['Home'].apply(clean_team_name_fbref)
  df_all['Away'] = df_all['Away'].apply(clean_team_name_fbref)
  df_all['date_scraped'] = df_all['date_scraped'].astype(str)

  try:
    df_all['Wk'] = pd.to_numeric(df_all['Wk'], errors='coerce')
    df_all['Wk'].fillna(0, inplace=True)  # Or drop the rows: df.dropna(subset=['Wk'], inplace=True)
  except Exception as e:
    print(f'fb ref failure {e}')

  
  # Attempt to convert the 'Attendance' column to numeric, coercing errors
  df_all['Attendance'] = pd.to_numeric(df_all['Attendance'], errors='coerce')
  
  # Now, you can proceed with creating a parquet table
  table = pa.Table.from_pandas(df_all)
  pq.write_table(table, './data/fb_ref_data.parquet', compression='BROTLI')
  print('fb ref data written')


## to do -- update this to only run once or a few times per day

# if datetime.now().hour in (2,12,20):
print('refreshing fb ref')
# df_soccer=pd.read_parquet('./data/df_soccer.parquet', engine='pyarrow')
# refresh_fbref_data(df_soccer)

df_soccer = pd.read_parquet('./data/fb_ref_data.parquet', engine='pyarrow')
refresh_fbref_data_updated(df_soccer)


# except Exception as e:
#   print(f'fbref failed -- {e}')

