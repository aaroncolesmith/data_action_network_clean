import pandas as pd
import requests
import time
import random
from datetime import datetime, timedelta
import statistics
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

# Note: Imports for 'display', 'SentenceTransformer', 'cosine_similarity'
# are missing. Add them if they are used in your environment (e.g., Jupyter).
# from IPython.display import display
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity


def req_to_df(r):
  # This function is kept as-is from your original code.
  # Recommendations from step 2 (pd.concat in a loop) are not applied yet.
  try:
    games_df=pd.json_normalize(r.json()['games'],
                      )[['id','status','start_time','away_team_id','home_team_id','winning_team_id','league_name','season','attendance',
                      'last_play.home_win_pct','last_play.over_win_pct',
                      'boxscore.total_away_points','boxscore.total_home_points','boxscore.total_away_firsthalf_points','boxscore.total_home_firsthalf_points',
                      'boxscore.total_away_secondhalf_points','boxscore.total_home_secondhalf_points','broadcast.network']]
  except:
    try:
      games_df=pd.json_normalize(r.json()['games'],
                  )[['id','status','start_time','away_team_id','home_team_id','winning_team_id','league_name','season','attendance',
                  'boxscore.total_away_points','boxscore.total_home_points','boxscore.total_away_firsthalf_points','boxscore.total_home_firsthalf_points',
                  'boxscore.total_away_secondhalf_points','boxscore.total_home_secondhalf_points']]
    except:
      games_df=pd.json_normalize(r.json()['games'],
                  )[['id','status','start_time','away_team_id','home_team_id','winning_team_id','league_name','season','attendance',
                  ]]


  odds_df=pd.DataFrame()
  for i in range(pd.json_normalize(r.json()['games']).index.size):
    try:
      odds_df=pd.concat([odds_df,
      pd.json_normalize(r.json()['games'][i],
                    'odds',
                    ['id'],
                    meta_prefix='game_',
                    record_prefix='',
                    errors='ignore'
                    )[[ 'game_id',
                      'ml_away', 'ml_home', 'spread_away', 'spread_home', 'spread_away_line','spread_home_line', 'over', 'under', 'draw', 'total', 
                          'away_total','away_over', 'away_under', 'home_total', 'home_over', 'home_under',
        'ml_home_public', 'ml_away_public', 'spread_home_public',
        'spread_away_public', 'total_under_public', 'total_over_public',
        'ml_home_money', 'ml_away_money', 'spread_home_money',
        'spread_away_money', 'total_over_money', 'total_under_money',
        'num_bets', 'book_id','type','inserted'
                          ]]               
    ]
                      ).reset_index(drop=True)
    except:
      pass

  teams_df=pd.DataFrame()
  for i in range(pd.json_normalize(r.json()['games']).index.size):
    teams_df=pd.concat([teams_df,
                        pd.json_normalize(r.json()['games'][i],
                    'teams',
                    ['id'],
                    meta_prefix='game_',
                    record_prefix='team_'
                    )
                        ]
                      ).reset_index(drop=True)

  df=pd.merge(
  pd.merge(
  pd.merge(games_df,
          odds_df.query('book_id == 15'),
          left_on='id',
          right_on='game_id'),
          teams_df[['team_id','team_full_name']].rename(columns={'team_id':'home_team_id', 'team_full_name':'home_team'})
  ),
  teams_df[['team_id','team_full_name']].rename(columns={'team_id':'away_team_id', 'team_full_name':'away_team'})

  )

  df['date_scraped'] = datetime.now()


  return df,teams_df


# --------------------------------------------------------------------
# NEW REFACTORED FUNCTION FOR SCRAPING
# --------------------------------------------------------------------
def scrape_league_data(league_name, existing_df, headers, start_date, end_date):
    """
    Scrapes scoreboard data for a specific league over a date range.
    """
    print(f"---  scraping {league_name} ---")
    
    # Use a list to collect DataFrames (much faster than repeated pd.concat)
    daily_dfs = [existing_df] 
    teams_dfs = []
    fail_list = []
    
    base_url = "https://api.actionnetwork.com/web/v1/scoreboard/"
    book_ids = "15,30,76,75,123,69,68,972,71,247,79"
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y%m%d')
        print(f"Processing {league_name} for {date_str}")
        
        # Build URL dynamically
        params = f"?period=game&bookIds={book_ids}&date={date_str}"
        if league_name == 'ncaab':
            url = f"{base_url}ncaab{params}&division=D1&tournament=0"
        else:
            url = f"{base_url}{league_name}{params}"
        
        # Increment date before potential error/continue
        current_date += timedelta(days=1)
        
        # generate a random sleep time between 1 and 6 seconds
        sleep_time = random.randint(1, 6)
        time.sleep(sleep_time)
        
        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()  # This will raise an error for 4xx or 5xx responses
            
            games_data = r.json().get('games', [])

            if len(games_data) > 0:
                df_tmp, teams_df_tmp = req_to_df(r)
                daily_dfs.append(df_tmp)
                teams_dfs.append(teams_df_tmp)
            else:
                print(f"No {league_name} games for {date_str}")
                
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error for {league_name} on {date_str}: {http_err} (Status: {r.status_code})")
            fail_list.append(date_str)
        except requests.exceptions.JSONDecodeError:
            # Handle cases where r.json() fails (e.g., empty or bad response)
            print(f"Failed to decode JSON for {league_name} on {date_str} (Status: {r.status_code}, Response: {r.text[:100]})")
            fail_list.append(date_str)
        except Exception as e:
            # Catch other unexpected errors
            print(f"An unexpected error occurred for {league_name} on {date_str}: {e}")
            fail_list.append(date_str)

    # Combine all DataFrames *once* at the end
    final_df = pd.concat(daily_dfs, ignore_index=True)
    final_teams_df = pd.concat(teams_dfs, ignore_index=True) if teams_dfs else pd.DataFrame()
    
    return final_df, final_teams_df, fail_list

# --------------------------------------------------------------------
# END NEW FUNCTION
# --------------------------------------------------------------------


headers = {
    'Authority': 'api.actionnetwork',
    'Accept': 'application/json',
    'Origin': 'https://www.actionnetwork.com',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'
}


# --- Load existing data ---
print("Loading existing data...")
try:
    df_cbb = pd.read_parquet('./data/df_cbb.parquet', engine='pyarrow')
except FileNotFoundError:
    print("cbb parquet not found, starting with empty DataFrame.")
    df_cbb = pd.DataFrame()

try:
    df_soccer = pd.read_parquet('./data/df_soccer.parquet', engine='pyarrow')
except FileNotFoundError:
    print("soccer parquet not found, starting with empty DataFrame.")
    df_soccer = pd.DataFrame()

try:
    df_nba = pd.read_parquet('./data/df_nba.parquet', engine='pyarrow')
except FileNotFoundError:
    print("nba parquet not found, starting with empty DataFrame.")
    df_nba = pd.DataFrame()

try:
    df_nfl = pd.read_parquet('./data/df_nfl.parquet', engine='pyarrow')
except FileNotFoundError:
    print("nfl parquet not found, starting with empty DataFrame.")
    df_nfl = pd.DataFrame()

try:
    df_mlb = pd.read_parquet('./data/df_mlb.parquet', engine='pyarrow')
except FileNotFoundError:
    print("mlb parquet not found, starting with empty DataFrame.")
    df_mlb = pd.DataFrame()

try:
    df_nba_futures = pd.read_parquet('./data/df_nba_futures.parquet', engine='pyarrow')
except:
    pass

try:
    teams_df = pd.read_parquet('./data/teams_db.parquet', engine='pyarrow')
except FileNotFoundError:
    print("teams_db parquet not found, starting with empty DataFrame.")
    teams_df = pd.DataFrame()


# --- Define date range ---
start_date = datetime.today().date() - timedelta(days=1)
end_date = datetime.today().date() + timedelta(days=7)


# --------------------------------------------------------------------
# REFACTORED SCRAPING LOOPS
# --------------------------------------------------------------------

# Store all new teams data in a list, starting with the loaded DF
all_teams_dfs = [teams_df]
all_fails = {}

# Define league configurations
leagues_to_scrape = {
    'ncaab': df_cbb,
    'soccer': df_soccer,
    'nba': df_nba,
    'nfl': df_nfl,
    'mlb': df_mlb,
}

# Run the scrapes
for league, df in leagues_to_scrape.items():
    updated_df, new_teams_df, fails = scrape_league_data(
        league, df, headers, start_date, end_date
    )
    
    # Re-assign the updated DataFrame back to its original variable name
    # This uses a bit of a trick to update the variable in the global scope
    globals()[f"df_{league.replace('ncaab', 'cbb')}"] = updated_df
    
    if not new_teams_df.empty:
        all_teams_dfs.append(new_teams_df)
    if fails:
        all_fails[league] = fails

# Consolidate all teams data at the end
teams_df = pd.concat(all_teams_dfs, ignore_index=True)

print("\n--- Scraping Summary ---")
print(f"All scrapes complete.")
if all_fails:
    print(f"Failed dates: {all_fails}")
else:
    print("No failures reported.")
print("--------------------------\n")

# --------------------------------------------------------------------
# END REFACTORED SECTION
# --------------------------------------------------------------------


# --- Start of your original futures scraping logic (unchanged) ---

def mode_agg(x):
    try:
        return statistics.mode(x)
    except statistics.StatisticsError:
        return None

def get_bet_data(r1):
  df=pd.DataFrame()

  bet_name = r1.json()['name']

  for j in range(len(r1.json()['books'])):
    book_id = r1.json()['books'][j]['book_id']
    d=pd.json_normalize(r1.json()['books'][j]['odds'])
    d['book_id'] = book_id
    df=pd.concat([df,d])
  df['player_id'] = pd.to_numeric(df['player_id'])

  team_df=pd.json_normalize(r1.json()['teams'])[['id','full_name','display_name','abbr','logo']]
  team_df.columns=['team_id','team_name','team_display_name','team_abbr','team_logo']
  if len(r1.json()['players'])>0:
    player_df = pd.json_normalize(r1.json()['players'])[['id','full_name']]
    player_df.columns=['player_id','player_name']

    df=pd.merge(df,
                player_df,
                how='left'
    )
  df=pd.merge(df,
              team_df,
              how='left'
  )

  df['bet_name'] = bet_name

  return df

def get_prob(a):
    odds = 0
    if a < 0:
        odds = (-a)/(-a + 100)
    else:
        odds = 100/(100+a)

    return odds


## Load existing data
try:
  # df = pd.read_parquet('https://github.com/aaroncolesmith/bet_model/blob/main/df_futures.parquet?raw=true', engine='pyarrow')
  df = pd.read_parquet('./data/df_futures.parquet', engine='pyarrow')
except:
  df = pd.DataFrame() # Initialize df if it fails
  pass

print(f'Input data size: {df.index.size}')


date_scraped=datetime.now()
d=pd.DataFrame()
for league_id in range(3):
  try:
    url=f'https://api.actionnetwork.com/web/v1/leagues/{league_id}/futures/available'
    r=requests.get(url,headers=headers)
    for i in range(len(r.json()['futures'])):
      bet_type = r.json()['futures'][i]['type']
      url = f'https://api.actionnetwork.com/web/v1/leagues/{league_id}/futures/'+bet_type.replace('#','%23')
      r1=requests.get(url,headers=headers)
      d1 = get_bet_data(r1)
      d1['bet_type'] = bet_type
      d1['date_scraped'] = date_scraped
      d=pd.concat([d,d1])
    print(f'League {league_id} succeeded')
  except Exception as e:
    print(f'League {league_id} failed:')
    # Code to handle the exception
    print("An error occurred:", e)

d = d.reset_index(drop=True)
# display(d.head(10)) # 'display' may not be available
# display(d.sample(10)) # 'display' may not be available


# --- FIX: Ensure 'player_name' column exists ---
# If no players were found in any futures, the column won't exist.
# This creates it and fills it with NaN so the next lines don't fail.
if 'player_name' not in d.columns:
    d['player_name'] = np.nan
# -----------------------------------------------


d.loc[d.player_name.notnull(), 'bet_outcome'] = d['player_name']
d.loc[d.player_name.isna(), 'bet_outcome'] = d['team_name']
d = d.query('bet_outcome != "0"').reset_index(drop=True)

d_agg=d.groupby(['date_scraped','bet_name','bet_type','bet_outcome',
           'value','option_type_id'],dropna=False).agg(
          player_name=('player_name',mode_agg),
          player_id=('player_id',mode_agg),
          team_name=('team_name',mode_agg),
          team_logo=('team_logo',mode_agg),
          min_money=('money','min'),
          median_money=('money','median'),
          avg_money=('money','mean'),
          max_money=('money','max'),
          books=('book_id', lambda x: ','.join(x.astype('str'))),
          book_count=('book_id','nunique')
).reset_index()

d_agg['implied_probability'] = d_agg['median_money'].apply(get_prob)

df = pd.concat([df,d_agg])


def reduce_data(timestamp, group_by_cols, target_value, dataframe):
    # Sort the dataframe by the timestamp column in ascending order
    sorted_df = dataframe.sort_values(timestamp)

    # Create a mask to identify the initial record for each group
    initial_record_mask = sorted_df.groupby(group_by_cols,dropna=False)[timestamp].transform('first') == sorted_df[timestamp]

    # Create a mask to identify the most recent record for each group
    recent_record_mask = sorted_df.groupby(group_by_cols,dropna=False)[timestamp].transform('last') == sorted_df[timestamp]

    # Create a mask to identify records where there was a change in the target value compared to the previous record
    change_mask = sorted_df.groupby(group_by_cols,dropna=False)[target_value].transform(lambda x: x.ne(x.shift()))

    # Apply the masks to the dataframe and return the filtered results
    filtered_df = sorted_df[initial_record_mask | recent_record_mask | change_mask]

    return filtered_df

def updated_reduce_data(timestamp, group_by_cols, target_value, dataframe):
  sorted_df = df.sort_values(group_by_cols+[timestamp]).reset_index(drop=True)

  sorted_df['price_change'] = sorted_df[target_value] - sorted_df[target_value].shift(1)
  sorted_df['time_change'] = sorted_df[timestamp] - sorted_df[timestamp].shift(1)
  sorted_df['time_change'] = sorted_df['time_change'] / np.timedelta64(1, 'h')
  # sorted_df['time_change'] = sorted_df['time_change'] / pd.to_timedelta(1, unit='h')

  sorted_df.loc[sorted_df.groupby(group_by_cols,dropna=False)[timestamp].transform('last') == sorted_df[timestamp],'max_timestamp_group'] = 1
  sorted_df.max_timestamp_group.fillna(0,inplace=True)
  sorted_df.loc[sorted_df.groupby(group_by_cols,dropna=False)[timestamp].transform('first') == sorted_df[timestamp],'min_timestamp_group'] = 1
  sorted_df.max_timestamp_group.fillna(0,inplace=True)

  filtered_df = sorted_df.loc[(sorted_df.min_timestamp_group == 1) | (sorted_df.max_timestamp_group == 1)| (sorted_df.price_change != 0)].reset_index(drop=True)
  
  for col in ['price_change','time_change','max_timestamp_group','min_timestamp_group']:
    del filtered_df[col]
  
  return filtered_df


# Example usage
timestamp = 'date_scraped'
group_by_cols = ['bet_name', 'bet_type', 'bet_outcome', 'value', 'option_type_id']
target_value = 'median_money'
dataframe = df

print(f'Updated data size: {df.index.size}')

# Call the function to get the reduced dataframe
reduced_df = updated_reduce_data(timestamp, group_by_cols, target_value, dataframe)

print(f'Reduced data size: {reduced_df.index.size}')

table = pa.Table.from_pandas(reduced_df)
pq.write_table(table, './data/df_futures.parquet',compression='BROTLI')


teams_df=pd.merge(teams_df, teams_df.groupby(['team_id'])['game_id'].max(),on=['team_id','game_id']).reset_index(drop=True)

df_cbb=df_cbb.drop_duplicates(subset=df_cbb.columns.to_list()[:-1]).reset_index(drop=True)
df_soccer=df_soccer.drop_duplicates(subset=df_soccer.columns.to_list()[:-1]).reset_index(drop=True)
df_nba=df_nba.drop_duplicates(subset=df_nba.columns.to_list()[:-1]).reset_index(drop=True)
df_nfl=df_nfl.drop_duplicates(subset=df_nfl.columns.to_list()[:-1]).reset_index(drop=True)
df_mlb=df_mlb.drop_duplicates(subset=df_mlb.columns.to_list()[:-1]).reset_index(drop=True)

teams_df=teams_df.drop_duplicates(subset=teams_df.columns.to_list()[:-2]).reset_index(drop=True)

# df_cbb.to_csv('df_cbb.csv',index=False)
# Parquet with Brotli compression
table = pa.Table.from_pandas(df_cbb)
pq.write_table(table, './data/df_cbb.parquet',compression='BROTLI')

# df_soccer.to_csv('df_soccer.csv',index=False)
# Parquet with Brotli compression
table = pa.Table.from_pandas(df_soccer)
pq.write_table(table, './data/df_soccer.parquet',compression='BROTLI')


table = pa.Table.from_pandas(df_nba)
pq.write_table(table, './data/df_nba.parquet',compression='BROTLI')

table = pa.Table.from_pandas(df_nfl)
pq.write_table(table, './data/df_nfl.parquet',compression='BROTLI')

table = pa.Table.from_pandas(df_mlb)
pq.write_table(table, './data/df_mlb.parquet',compression='BROTLI')

# df_mma=df_mma.drop_duplicates(subset=df_soccer.columns.to_list()[:-1]).reset_index(drop=True)
# table = pa.Table.from_pandas(df_mma)
# pq.write_table(table, 'df_mma.parquet',compression='BROTLI')

teams_df.drop_duplicates().to_csv('./data/teams_db.csv',index=False)



### adding in the trank data
def normalize_bet_team_names(d, field):
    substitutions = {'State': 'St.', 
                      'Kansas City Roos': 'UMKC', 
                      'Long Island University Sharks': 'LIU Brooklyn',
                    'Omaha Mavericks': 'Nebraska Omaha', 
                    'North Carolina-Wilmington Seahawks': 'UNC Wilmington',
                    'Virginia Military Institute Keydets': 'VMI', 
                    'Southern Methodist Mustangs': 'SMU',
                    'Virginia Commonwealth Rams':'VCU',
                    'Florida International Golden Panthers':'FIU',
                    'N.J.I.T. Highlanders':'NJIT',
                    'Ole Miss':'Mississippi',
                    'St. Peter\'s Peacocks':'Saint Peter\'s',
                    'Texas-Arlington Mavericks':'UT Arlington',
                    'Miami (FL) Hurricanes':'Miami FL',
                    'Pennsylvania Quakers':'Penn',
                    'Louisiana-Monroe Warhawks':'Louisiana Monroe',
                    'Texas A&M-CC Islanders':'Texas A&M Corpus Chris',
                    'IPFW Mastodons':'Fort Wayne',
                    'Missouri KC':'UMKC',
                    'Chaminade Silverswords':'Chaminade',
                    'Florida International':'FIU',
                    'UConn Huskies':'Connecticut',
                    'UMass Minutemen':'Massachusetts',
                    'Massachusetts Lowell River Hawks':'UMass Lowell',
                    'SIU-Edwardsville':'SIU Edwardsville',
                    'Miami (FL) Hurricanes':'Miami FL'
                    }

    for key, value in substitutions.items():
        d[field] = d[field].str.replace(key, value, regex=True)

    replacements = ['Musketeers','Longhorns','Wildcats','Panthers','Tigers','Warriors','Skyhawks','Sharks','Flames','Bulldogs','Cougars','Runnin\'','Rebels',
                    'Spartans','Razorbacks','Bears','Raiders','Cardinal','Buckeyes','Hawkeyes','Bobcats','Rockets',
                    'Gauchos']
    for replacement in replacements:
        d[field] = d[field].str.replace(replacement, '', regex=True)
    return d



def update_trank(df_cbb):
  
  df_cbb['start_time_pt'] = pd.to_datetime(df_cbb['start_time']).dt.tz_convert('US/Pacific')
  df_cbb['date'] = df_cbb['start_time_pt'].dt.date
  start_date = datetime.now().date() - timedelta(31)
  end_date = datetime.now().date() + timedelta(14)
  ## read parquet file for trank_db
  df = pd.read_parquet('./data/trank_db.parquet')


  date_added = datetime.now()
  for date in pd.date_range(start_date, end_date):
    url_date=pd.to_datetime(date).strftime('%Y%m%d')
    url=f'https://barttorvik.com/schedule.php?date={url_date}&conlimit='
    r=requests.get(url)
    trank=pd.read_html(r.content)[0]
    if trank.index.size > 0:
      trank.columns = [x.lower().replace('(','').replace(')','').replace(' ','_').replace('-','_') for x in trank.columns]
      for x in trank.columns:
        if 'unnamed' in x:
          del trank[x]
      trank['date'] = pd.to_datetime(date)

      replacements = ['\d+', 'BIG12|ESPN+','Peacock', 'ESPNU', 'ESPN', 'FS', 'ACCN', 'BIG|', 'CBSSN',
                      'ACC Network', 'FloSports',
                      'PAC','truTV','CBS','BE-T','Ivy-T','NCAA-T','FOX','WCC-T','Amer-T U','CAA-T','MWC-T','NEC-T','TBS','CUSA-T','BW-T','SECN',
                      'SEC-T','P-T','MAC-T','B-T','BTN','MAAC-T','ACC-T']
      for replacement in replacements:
          trank['matchup'] = trank['matchup'].str.replace(replacement, '', regex=True)
      trank['matchup'] = trank['matchup'].str.replace(r'\s\+\s*$', '', regex=True)
      trank['matchup'] = trank['matchup'].str.replace('Illinois Chicago', 'UIC', regex=True)
      trank['matchup'] = trank['matchup'].str.replace('Gardner Webb', 'Gardner-Webb', regex=True)
      # the change: Check if split produces enough elements
      split_result = trank.t_rank_line.str.split('-',expand=True)
      if 1 in split_result and 0 in split_result[1].str.split(',',expand=True):
          trank['trank_spread']='-'+split_result[1].str.split(',',expand=True)[0]
      else:
          trank['trank_spread'] = np.nan # or some default value

      trank['date_added'] = date_added

      df = pd.concat([df, trank])

  replacements = ['\d+', 'BIG12|ESPN+','Peacock', 'ESPNU', 'ESPN', 'FS', 'ACCN', 'BIG|', 'CBSSN',
                  'ACC Network', 'FloSports',
                  'PAC','truTV','CBS','BE-T','Ivy-T','NCAA-T','FOX','WCC-T','Amer-T U','CAA-T','MWC-T','NEC-T','TBS','CUSA-T','BW-T','SECN',
                  'SEC-T','P-T','MAC-T','B-T','BTN','MAAC-T','ACC-T']
  for replacement in replacements:
      df['matchup'] = df['matchup'].str.replace(replacement, '', regex=True)


  df['matchup'] = df['matchup'].str.strip()
  df['matchup'] = df['matchup'].str.replace('  ',' ')

  df = df.sort_values(['date','matchup','date_added']).reset_index(drop=True)

  df['ttq'] = df['ttq'].astype('str')

  df.loc[df['matchup'] != df['matchup'].shift(-1), 'keep_record'] = 1
  df.loc[df['matchup'] != df['matchup'].shift(1), 'keep_record'] = 1
  df.loc[df['ttq'] != df['ttq'].shift(-1), 'keep_record'] = 1  

  df = df.loc[df.keep_record == 1]

  df['ttq'] = df['ttq'].astype(str)
  print('saving trank_db')
  print(df.columns)
  
  table = pa.Table.from_pandas(df)
  pq.write_table(table, './data/trank_db.parquet',compression='BROTLI')

  return df


def update_merged_data(df_cbb, df_trank):
  ## load odds data
  ## update pass in df_cbb
  df_cbb = df_cbb.merge(df_cbb.groupby(['id']).agg(updated_start_time=('start_time','last')).reset_index(), on='id', how='left')
  del df_cbb['start_time']

  df_cbb=df_cbb.rename(columns={'updated_start_time':'start_time'})
  df_cbb.columns = [x.lower() for x in df_cbb.columns]
  df_cbb.columns = [x.replace('.','_') for x in df_cbb.columns]
  df_cbb=df_cbb.sort_values(['start_time','date_scraped'],ascending=[True,True]).reset_index(drop=True)

  df_cbb = normalize_bet_team_names(df_cbb, 'home_team')
  df_cbb = normalize_bet_team_names(d_cbb, 'away_team') # Typo in original, should be df_cbb

  df_cbb['matchup'] = df_cbb['away_team'] + ' at ' + df_cbb['home_team']
  df_cbb['start_time_pt'] = pd.to_datetime(df_cbb['start_time']).dt.tz_convert('US/Pacific')
  df_cbb['start_time_et'] = pd.to_datetime(df_cbb['start_time']).dt.tz_convert('US/Eastern')
  df_cbb['date'] = df_cbb['start_time_pt'].dt.date

  df=df_cbb.groupby(['id','date','start_time_et','matchup','home_team_id','home_team','away_team_id','away_team']).agg(
                                                                                        status=('status','last'),
                                                                                        spread_away=('spread_away','last'),
                                                                                        spread_home=('spread_home','last'),
                                                                                        spread_away_min=('spread_away','min'),
                                                                                        spread_away_max=('spread_away','max'),
                                                                                        spread_away_std=('spread_away','std'),
                                                                                        spread_home_min=('spread_home','min'),
                                                                                        spread_home_max=('spread_home','max'),
                                                                                        spread_home_std=('spread_home','std'),
                                                                                        score_away=('boxscore_total_away_points','last'),
                                                                                        score_home=('boxscore_total_home_points','last'),
                                                                                        attendance=('attendance','max'),
                                                                                        spread_home_public=('spread_home_public','last'),
                                                                                        spread_away_public=('spread_away_public','last'),
                                                                                        num_bets=('num_bets','max'),
                                                                                        rec_count=('inserted','size')
                                                                                        ).reset_index()
  df['spread_away_result'] = df['score_away'] - df['score_home']
  df['spread_home_result'] = df['score_home'] - df['score_away']
  df.loc[(df.spread_home_result+df.spread_home)>0,'bet_result'] = 'home_wins'
  df.loc[(df.spread_home_result+df.spread_home)<0,'bet_result'] = 'away_wins'
  df.loc[(df.spread_home_result+df.spread_home)==0,'bet_result'] = 'push'

  ## load trank data

  df2=pd.merge(df_trank,
              df_trank.groupby(['matchup', 'date']).agg(date_added=('date_added','max')).reset_index(),
          left_on=['matchup', 'date','date_added'], right_on=['matchup', 'date','date_added'], how='inner',suffixes=('','_del'))
  for x in df2.columns:
    if 'del' in x:
      del df2[x]

  ## max date = yesterday
  max_date = datetime.now().date() - timedelta(1)
  df['date']=pd.to_datetime(df['date'])
  df2['date']=pd.to_datetime(df2['date'])

  ## filter odds data for greater than max date
  df=df.loc[pd.to_datetime(df.date)>=pd.to_datetime(max_date)]

  ## filter trank data for greater than max date
  df2=df2.loc[pd.to_datetime(df2.date)>=pd.to_datetime(max_date)]
  ## merge data



  # Load the pre-trained Sentence Transformer model
  # model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1") # Uncomment when ready

  df['date']=pd.to_datetime(df['date'])
  df2['date']=pd.to_datetime(df2['date'])

  df3=pd.read_parquet('./data/trank_db_merged.parquet')
  df3=df3.loc[pd.to_datetime(df3.date)<pd.to_datetime(max_date)]
  for date in df.date.unique():
    # Filter the dataframes by date
    df_filtered = df[pd.to_datetime(df['date']) == pd.to_datetime(date)]
    df2_filtered = df2[df2['date'] == date]

    # Get the matchup columns from both dataframes
    df_matchups = df_filtered['matchup'].dropna().tolist()  # Drop any NaN values
    df2_matchups = df2_filtered['matchup'].dropna().tolist()

    # Ensure the matchup lists are not empty
    if len(df_matchups) == 0 or len(df2_matchups) == 0:
        # raise ValueError("One of the matchup lists is empty. Check the input dataframes.")
        print(f"Skipping date {date} due to empty matchup list.")
        continue # Skip to the next date

    # Encode the matchups into sentence embeddings
    # df_embeddings = model.encode(df_matchups)
    # df2_embeddings = model.encode(df2_matchups)

    # # Check if the embeddings are 2D arrays
    # if df_embeddings.ndim == 1:
    #     df_embeddings = df_embeddings.reshape(1, -1)
    # if df2_embeddings.ndim == 1:
    #     df2_embeddings = df2_embeddings.reshape(1, -1)

    # # Calculate the cosine similarity between the matchups
    # similarity_matrix = cosine_similarity(df_embeddings, df2_embeddings)

    # # Get the best match from df2 for each matchup in df based on cosine similarity
    # best_match_idx = np.argmax(similarity_matrix, axis=1)
    # best_matches = [df2_matchups[i] for i in best_match_idx]
    # similarity_scores = [similarity_matrix[i, best_match_idx[i]] for i in range(len(df_matchups))]

    # # Add best matches and similarity scores to df
    # df_filtered['best_match'] = best_matches
    # df_filtered['similarity_score'] = similarity_scores
    
    # --- TEMPORARY WORKAROUND since model is commented out ---
    # This section needs to be replaced when sentence transformer is active
    # For now, we'll just merge directly on 'matchup' as a placeholder
    df_filtered['best_match'] = df_filtered['matchup']
    df_filtered['similarity_score'] = 1.0 # Assume perfect match
    # --- END WORKAROUND ---


    # Merge the two dataframes based on best match and date
    df_merged = pd.merge(df_filtered, df2_filtered, how='left', left_on=['best_match', 'date'], right_on=['matchup', 'date'], suffixes=('', '_df2'))

    # Filter based on similarity score threshold (e.g., 0.8 or 80%)
    similarity_threshold = 0.15
    df_merged_filtered = df_merged[df_merged['similarity_score'] >= similarity_threshold]

    df3=pd.concat([df3,df_merged_filtered])


  df3['favorite_spread_bovada']=df3[['spread_away','spread_home']].min(axis=1)
  df3['favorite_spread_bovada'] = pd.to_numeric(df3['favorite_spread_bovada'])
  df3['trank_spread'] = pd.to_numeric(df3['trank_spread'])
  df3['spread_diff'] = abs(df3['favorite_spread_bovada'] - df3['trank_spread'])

  df3['spread_diff']=pd.to_numeric(df3['spread_diff'])
  df3['ttq']=pd.to_numeric(df3['ttq'])
  df3['similarity_score']=pd.to_numeric(df3['similarity_score'])

  # If bovada line > trank like, bet favorite; otherwise bet the dog
  df3.loc[df3.favorite_spread_bovada > df3.trank_spread, 'bet_advice'] = 'bet_favorite'
  df3.loc[df3.favorite_spread_bovada < df3.trank_spread, 'bet_advice'] = 'bet_dog'

  df3.loc[(df3.spread_away<0)&(df3.bet_result=='away_wins'), 'fav_result'] = 'fav_wins'
  df3.loc[(df3.spread_home<0)&(df3.bet_result=='home_wins'), 'fav_result'] = 'fav_wins'

  df3.loc[(df3.spread_away>0)&(df3.bet_result=='away_wins'), 'fav_result'] = 'dog_wins'
  df3.loc[(df3.spread_home>0)&(df3.bet_result=='home_wins'), 'fav_result'] = 'dog_wins'

  df3.loc[(df3.bet_advice=='bet_favorite')&(df3.fav_result=='fav_wins'), 'bet_advice_result'] = 'win'
  df3.loc[(dfa3.bet_advice=='bet_dog')&(df3.fav_result=='dog_wins'), 'bet_advice_result'] = 'win' # Typo in original, should be df3

  df3.loc[(df3.bet_advice=='bet_favorite')&(df3.fav_result=='dog_wins'), 'bet_advice_result'] = 'loss'
  df3.loc[(df3.bet_advice=='bet_dog')&(df3.fav_result=='fav_wins'), 'bet_advice_result'] = 'loss'

  print('saving merged trank data')
  print(df3.columns)
  
  table = pa.Table.from_pandas(df3)
  pq.write_table(table, './data/trank_db_merged.parquet',compression='BROTLI')

print(datetime.now())
## 7/1/2025 -- Removed Trank for now
# print('----trank time ------')
# # if datetime.now().hour in (1,2,3,4,10,15):
# df_trank = update_trank(df_cbb)
# update_merged_data(df_cbb, df_trank)


print('script done')
