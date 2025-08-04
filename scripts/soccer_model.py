import pandas as pd
import numpy as np
import re
from datetime import datetime
import pyarrow
import pyarrow as pa
import pyarrow.parquet as pq


def get_prob(a):
    odds = 0
    if a < 0:
        odds = (-a)/(-a + 100)
    else:
        odds = 100/(100+a)

    return odds

def fav_payout(ml):
    try:
        return 100 / abs(ml)
    except ZeroDivisionError:
        return None

def dog_payout(ml):
  return ml/100

df=pd.read_parquet('https://github.com/aaroncolesmith/data_action_network/raw/refs/heads/main/data/df_soccer.parquet', engine='pyarrow')

df['home_score'] = df['boxscore.total_home_points']
df['away_score'] = df['boxscore.total_away_points']

df['total_score'] = df['home_score'] + df['away_score']

df['under_implied_probability'] = df['under'].apply(get_prob)
df['over_implied_probability'] = df['over'].apply(get_prob)

df['date_scraped'] = pd.to_datetime(df['date_scraped'])

df = df.sort_values(['id','start_time','date_scraped'],ascending=[True,True,True]).reset_index(drop=True)

# Perform the aggregation
df_agg = df.groupby(['home_team_id']).agg(
    most_recent_name=('home_team', 'first'),
    unique_names=('home_team', 'nunique'),
    unique_name_list=('home_team', lambda x: ', '.join(set(x)))
).sort_values('unique_names', ascending=False).reset_index()

if len(df_agg.loc[df_agg.unique_names >1]) > 0:
  # Create the dictionary: home_team_id as the key, most_recent_name as the value
  id_name_dict = df_agg.set_index('home_team_id')['most_recent_name'].to_dict()

  # Assuming id_name_dict is the dictionary created earlier
  # Update home_team based on home_team_id
  df['home_team'] = df['home_team_id'].map(id_name_dict).fillna(df['home_team'])

  # Update away_team based on away_team_id
  df['away_team'] = df['away_team_id'].map(id_name_dict).fillna(df['away_team'])

del df_agg


d = df.groupby(['id']).agg(
    start_time=('start_time','last'),
    status=('status','last'),
    home_team=('home_team','last'),
    home_team_id=('home_team_id','last'),
    away_team=('away_team','last'),
    away_team_id=('away_team_id','last'),
    league_name=('league_name','last'),
    season=('season','last'),
    home_score=('home_score','last'),
    away_score=('away_score','last'),
    total_score=('total_score','last'),
    ml_home=('ml_home','last'),
    ml_home_first=('ml_home','first'),
    ml_home_avg=('ml_home','mean'),
    ml_away=('ml_away','last'),
    ml_away_first=('ml_away','first'),
    ml_away_avg=('ml_away','mean'),
    draw=('draw','last'),
    draw_first=('draw','first'),
    draw_avg=('draw','mean'),
    spread_home=('spread_home','last'),
    spread_home_first=('spread_home','first'),
    spread_home_avg=('spread_home','mean'),
    spread_away=('spread_away','last'),
    spread_away_first=('spread_away','first'),
    spread_away_avg=('spread_away','mean'),
    spread_home_line=('spread_home_line','last'),
    spread_home_line_first=('spread_home_line','first'),
    spread_home_line_avg=('spread_home_line','mean'),
    spread_away_line=('spread_away_line','last'),
    spread_away_line_first=('spread_away_line','first'),
    spread_away_line_avg=('spread_away_line','mean'),
    over=('over','last'),
    over_first=('over','first'),
    over_avg=('over','mean'),
    over_implied_probability=('over_implied_probability','last'),
    over_implied_probability_first=('over_implied_probability','first'),
    over_implied_probability_avg=('over_implied_probability','mean'),
    under=('under','last'),
    under_first=('under','first'),
    under_avg=('under','mean'),
    under_implied_probability=('under_implied_probability','last'),
    under_implied_probability_first=('under_implied_probability','first'),
    under_implied_probability_avg=('under_implied_probability','mean'),
    total=('total','last'),
    total_first=('total','first'),
    total_avg=('total','mean'),
    home_total=('home_total','last'),
    home_total_first=('home_total','first'),
    home_total_avg=('home_total','mean'),
    away_total=('away_total','last'),
    away_total_first=('away_total','first'),
    away_total_avg=('away_total','mean'),
    ml_home_public=('ml_home_public','last'),
    ml_home_public_first=('ml_home_public','first'),
    ml_home_public_avg=('ml_home_public','mean'),
    ml_away_public=('ml_away_public','last'),
    ml_away_public_first=('ml_away_public','first'),
    ml_away_public_avg=('ml_away_public','mean'),
    ml_home_money=('ml_home_money','last'),
    ml_home_money_first=('ml_home_money','first'),
    ml_home_money_avg=('ml_home_money','mean'),
    ml_away_money=('ml_away_money','last'),
    ml_away_money_first=('ml_away_money','first'),
    ml_away_money_avg=('ml_away_money','mean'),
    num_bets=('num_bets','last'),
    num_bets_first=('num_bets','first'),
    num_bets_avg=('num_bets','mean'),
    num_bets_median=('num_bets','median'),
    num_bets_std=('num_bets','std'),
).reset_index()


d['start_time_pt'] = pd.to_datetime(d['start_time']).dt.tz_convert('US/Pacific')
d['date'] = pd.to_datetime(d['start_time_pt']).dt.date

d['home_covered_spread'] = np.where((d['home_score'] + d['spread_home']) > d['away_score'], 1, 0)
d['away_covered_spread'] = np.where((d['away_score'] + d['spread_away']) > d['home_score'], 1, 0)

d.loc[d['spread_home_line'] < 0, 'spread_home_payout'] = d['spread_home_line'].apply(fav_payout)*d['home_covered_spread']
d.loc[d['spread_home_line'] > 0, 'spread_home_payout'] = d['spread_home_line'].apply(dog_payout)*d['home_covered_spread']
d.loc[d['home_covered_spread'] == 0, 'spread_home_payout'] = -1

d.loc[d['spread_away_line'] < 0, 'spread_away_payout'] = d['spread_away_line'].apply(fav_payout)*d['away_covered_spread']
d.loc[d['spread_away_line'] > 0, 'spread_away_payout'] = d['spread_away_line'].apply(dog_payout)*d['away_covered_spread']
d.loc[d['away_covered_spread'] == 0, 'spread_away_payout'] = -1


# Determine if the total score is under/over the betting total
d['under_hit'] = np.where(d['total_score'] < d['total'], 1,
                    np.where(d['total_score'] > d['total'], 0, np.nan))

d['over_hit'] = np.where(d['total_score'] > d['total'], 1,
                    np.where(d['total_score'] < d['total'], 0, np.nan))


d.loc[d['under'] < 0, 'under_payout'] = d['under'].apply(fav_payout)*d['under_hit']
d.loc[d['under'] > 0, 'under_payout'] = d['under'].apply(dog_payout)*d['under_hit']
d.loc[d['under_hit'] == 0, 'under_payout'] = -1


d.loc[d['over'] < 0, 'over_payout'] = d['over'].apply(fav_payout)*d['over_hit']
d.loc[d['over'] > 0, 'over_payout'] = d['over'].apply(dog_payout)*d['over_hit']
d.loc[d['over_hit'] == 0, 'over_payout'] = -1

df_fb_ref = pd.read_parquet('https://github.com/aaroncolesmith/data_action_network/raw/refs/heads/main/data/fb_ref_data.parquet', engine='pyarrow')
print(df_fb_ref.index.size)
df_fb_ref.drop_duplicates(subset=['url'], keep='last', inplace=True)
print(df_fb_ref.index.size)
## filter df_fb_ref for any Round not in female_leagues list
female_leagues = ['Women\'s Super League','Liga F']
df_fb_ref = df_fb_ref[~df_fb_ref['Round'].isin(female_leagues)]
df_fb_ref = df_fb_ref[~df_fb_ref['url'].str.contains('Frauen-Bundes', na=False)]
df_fb_ref = df_fb_ref[~df_fb_ref['url'].str.contains('Premiere-Ligue', na=False)]

df_team_map = pd.read_parquet('data/team_mapping_data.parquet')
d = pd.merge(d,df_team_map[['team','team_fbref']],left_on='home_team',right_on='team',how='left').rename(columns={'team':'del_team_1','team_fbref':'home_team_fbref'})
d = pd.merge(d,df_team_map[['team','team_fbref']],left_on='away_team',right_on='team',how='left').rename(columns={'team':'del_team_2','team_fbref':'away_team_fbref'})

for col in d.columns:
  if 'del' in col:
    del d[col]

d['date'] = pd.to_datetime(d['date'])
df_fb_ref['date'] = pd.to_datetime(df_fb_ref['date'])
d = pd.merge(d,df_fb_ref,left_on=['date','home_team_fbref','away_team_fbref'],right_on=['date','Home','Away'],how='left')

for col in ['Home','Away','Match Report','date_scraped']:
  del d[col]

def calculate_rolling_metrics_avg(dataframe, team_id_col, metrics, windows):
        """
        Calculate rolling metrics for different window sizes

        Args:
            dataframe (pd.DataFrame): Input dataframe
            team_id_col (str): Column name for team ID
            metrics (list): List of metrics to calculate
            windows (list): List of window sizes

        Returns:
            pd.DataFrame: Dataframe with added rolling metrics
        """
        result_df = dataframe.copy()

        for metric in metrics:
            for window in windows:
                # Game-based rolling window
                result_df[f'{metric}_last_{window}'] = (
                    result_df.groupby(team_id_col)[metric]
                    .transform(lambda x: x.shift(fill_value=0).rolling(window=window, min_periods=1).sum()) /
                    result_df.groupby(team_id_col)[metric]
                    .transform(lambda x: x.shift(fill_value=0).rolling(window=window, min_periods=1).count())
                )


        return result_df

def calculate_rolling_metrics_sum(dataframe, team_id_col, metrics, windows):
        """
        Calculate rolling metrics for different window sizes

        Args:
            dataframe (pd.DataFrame): Input dataframe
            team_id_col (str): Column name for team ID
            metrics (list): List of metrics to calculate
            windows (list): List of window sizes

        Returns:
            pd.DataFrame: Dataframe with added rolling metrics
        """
        result_df = dataframe.copy()

        for metric in metrics:
            for window in windows:
                # Game-based rolling window
                result_df[f'{metric}_last_{window}_sum'] = (
                    result_df.groupby(team_id_col)[metric]
                    .transform(lambda x: x.shift(fill_value=0).rolling(window=window, min_periods=1).sum())
                )


        return result_df

# Prepare data for rolling metrics calculation
d_home = d[['id', 'status', 'start_time', 'away_team_id', 'away_team', 'under_hit',
       'total_score', 'away_score', 'away_xg', 'home_xg']].rename(columns={
           'away_team': 'team',
           'away_team_id': 'team_id',
           'away_score': 'team_score',
           'away_xg': 'xg',
           'home_xg': 'opp_xg'
       })
d_home['home'] = 1

d_away = d[['id', 'status', 'start_time', 'home_team_id', 'home_team', 'under_hit',
       'total_score', 'home_score', 'home_xg', 'away_xg']].rename(columns={
           'home_team': 'team',
           'home_team_id': 'team_id',
           'home_score': 'team_score',
           'home_xg': 'xg',
           'away_xg': 'opp_xg'
       })
d_away['home'] = 0

d3 = pd.concat([
    d_home,
    d_away
]).sort_values(['start_time', 'id'], ascending=[True, True])

# Calculate opponent scores
d3['opp_score'] = d3['total_score'] - d3['team_score']
d3['win'] = np.where(d3['team_score'] > d3['opp_score'], 1, 0)
d3['loss'] = np.where(d3['team_score'] < d3['opp_score'], 1, 0)
d3['tie'] = np.where(d3['team_score'] == d3['opp_score'],1,0)

d3_initial_columns = d3.columns.tolist()


# Define metrics and windows for rolling calculations
metrics = ['under_hit', 'total_score', 'team_score', 'opp_score', 'xg', 'opp_xg', 'win','loss','tie']
windows = [3, 5, 10, 15, 25, 30, 50]

# Calculate rolling metrics
d3 = calculate_rolling_metrics_avg(d3, 'team_id', metrics, windows)

metrics = ['win','loss','tie']
d3 = calculate_rolling_metrics_sum(d3, 'team_id', metrics, windows)
d3_post_columns = d3.columns.tolist()

new_d3_cols = list(set(d3_post_columns) - set(d3_initial_columns))

d = pd.merge(
    d,
    d3[["id", "team_id"] + new_d3_cols],
    how="left",
    left_on=["id", "home_team_id"],
    right_on=["id", "team_id"],
    suffixes=("", "_del"),
)

for col in new_d3_cols:
    d[f"home_{col}"] = d[col]
    try:
        del d[col]
    except:
        pass

d = pd.merge(
    d,
    d3[["id", "team_id"] + new_d3_cols],
    how="left",
    left_on=["id", "away_team_id"],
    right_on=["id", "team_id"],
)

for col in new_d3_cols:
    d[f"away_{col}"] = d[col]
    try:
        del d[col]
    except:
        pass
    pass

    ########

d["odds_adjusted_total"] = d["total"] * ((1 - d["under_implied_probability"]) + 0.5)

d["combined_total_score_last_10"] = (
    d["home_total_score_last_10"] + d["away_total_score_last_10"]
) / 2
d["total_score_ratio"] = d["combined_total_score_last_10"] / d["total"]

# Add rolling over/under trends by league
def add_market_reaction_features(df):
    # Last 7 days over rate by league
    df = df.sort_values('start_time')
    for league in df['league_name'].unique():
        league_mask = df['league_name'] == league
        df.loc[league_mask, 'recent_over_rate'] = (
            df.loc[league_mask, 'over_hit']
            .rolling(window=7, min_periods=3)
            .mean()
            .shift(1)  # Critical - only use past information
        )

    # Bookmaker reaction feature
    df['odds_reaction'] = df['over_implied_probability_avg'] - df['recent_over_rate']

    # Streak features
    df['over_streak'] = df.groupby('league_name')['over_hit'].transform(
        lambda x: x.rolling(5, min_periods=2).sum().shift(1)
    )

    return df

# Apply to both training and test data
d = add_market_reaction_features(d)

d['home_score_diff_last_10'] = d['home_total_score_last_10'] - d['home_opp_score_last_10']
d['away_score_diff_last_10'] = d['away_total_score_last_10'] - d['away_opp_score_last_10']

d['home_score_diff_last_10_pct_diff'] = round((d['home_score_diff_last_10'] - d['away_score_diff_last_10']) / d['home_score_diff_last_10'],4).fillna(0)
d['home_score_diff_last_10_abs_diff'] = round((d['home_score_diff_last_10'] - d['away_score_diff_last_10']),4).fillna(0)
d['home_score_diff_last_10_major_advantage'] = np.where(d['home_score_diff_last_10_pct_diff']>= d['home_score_diff_last_10_pct_diff'].quantile(.8),1,0)
d['away_score_diff_last_10_major_advantage'] = np.where(d['home_score_diff_last_10_pct_diff']<= d['home_score_diff_last_10_pct_diff'].quantile(.2),1,0)

d['home_score_diff_last_5'] = d['home_total_score_last_5'] - d['home_opp_score_last_5']
d['away_score_diff_last_5'] = d['away_total_score_last_5'] - d['away_opp_score_last_5']

d['home_score_diff_last_5_pct_diff'] = round((d['home_score_diff_last_5'] - d['away_score_diff_last_5']) / d['home_score_diff_last_5'],4).fillna(0)
d['home_score_diff_last_5_abs_diff'] = round((d['home_score_diff_last_5'] - d['away_score_diff_last_5']),4).fillna(0)
d['home_score_diff_last_5_major_advantage'] = np.where(d['home_score_diff_last_5_pct_diff']>= d['home_score_diff_last_5_pct_diff'].quantile(.8),1,0)
d['away_score_diff_last_5_major_advantage'] = np.where(d['home_score_diff_last_5_pct_diff']<= d['home_score_diff_last_5_pct_diff'].quantile(.2),1,0)


d['home_score_diff_last_25'] = d['home_total_score_last_25'] - d['home_opp_score_last_25']
d['away_score_diff_last_25'] = d['away_total_score_last_25'] - d['away_opp_score_last_25']

d['home_score_diff_last_25_pct_diff'] = round((d['home_score_diff_last_25'] - d['away_score_diff_last_25']) / d['home_score_diff_last_25'],4).fillna(0)
d['home_score_diff_last_25_abs_diff'] = round((d['home_score_diff_last_25'] - d['away_score_diff_last_25']),4).fillna(0)
d['home_score_diff_last_25_major_advantage'] = np.where(d['home_score_diff_last_25_pct_diff']>= d['home_score_diff_last_25_pct_diff'].quantile(.8),1,0)
d['away_score_diff_last_25_major_advantage'] = np.where(d['home_score_diff_last_25_pct_diff']<= d['home_score_diff_last_25_pct_diff'].quantile(.2),1,0)

d['home_score_diff_last_50'] = d['home_total_score_last_50'] - d['home_opp_score_last_50']
d['away_score_diff_last_50'] = d['away_total_score_last_50'] - d['away_opp_score_last_50']

d['home_score_diff_last_50_pct_diff'] = round((d['home_score_diff_last_50'] - d['away_score_diff_last_50']) / d['home_score_diff_last_50'],4).fillna(0)
d['home_score_diff_last_50_abs_diff'] = round((d['home_score_diff_last_50'] - d['away_score_diff_last_50']),4).fillna(0)
d['home_score_diff_last_50_major_advantage'] = np.where(d['home_score_diff_last_50_pct_diff']>= d['home_score_diff_last_50_pct_diff'].quantile(.8),1,0)
d['away_score_diff_last_50_major_advantage'] = np.where(d['home_score_diff_last_50_pct_diff']<= d['home_score_diff_last_50_pct_diff'].quantile(.2),1,0)

# Momentum/Form Features
d["home_form_last5"] = (d["home_team_score_last_5"] - d["home_opp_score_last_5"])/2
d["away_form_last5"] = (d["away_team_score_last_5"] - d["away_opp_score_last_5"])/2
d["home_attack_strength"] = d["home_xg_last_5"] - d["home_opp_xg_last_5"]
d["away_attack_strength"] = d["away_xg_last_5"] - d["away_opp_xg_last_5"]

# Odds Movement Features
d["over_odds_change"] = d["over"] - d["over_first"]
d["odds_implied_prob_diff"] = d["over_implied_probability_avg"] - d["under_implied_probability_avg"]

# # Interaction Features
d["total_xg_product"] = d["home_xg_last_5"] * d["away_xg_last_5"]
d["odds_total_ratio"] = d["odds_adjusted_total"] / d["total_avg"]


# Momentum features
d["home_xg_momentum"] = d["home_xg_last_5"] / d["home_xg_last_10"]  # Recent vs longer-term form
d["away_xg_momentum"] = d["away_xg_last_5"] / d["away_xg_last_10"]  # Recent vs longer-term form

d["home_xg_defense_momentum"] = d["home_opp_xg_last_5"] / d["home_opp_xg_last_10"]  # Recent vs longer-term form
d["away_xg_defense_momentum"] = d["away_opp_xg_last_5"] / d["away_opp_xg_last_10"]  # Recent vs longer-term form

d["home_defensive_weakness"] = (d["home_opp_score_last_3"] - d["home_opp_xg_last_3"])/2  # Overperformance against xG
d["away_defensive_weakness"] = (d["away_opp_score_last_3"] - d["away_opp_xg_last_3"])/2  # Overperformance against xG

# Odds market movement
d["over_implied_prob_trend"] = (d["over_implied_probability_avg"]
                                 - d["over_implied_probability_first"])

# Match importance
d["home_league_position"] = d.groupby(["league_name", "season"])["home_team_score_last_50"].rank(ascending=False)
d["away_relegation_risk"] = d.groupby(["league_name", "season"])["away_team_score_last_50"].rank(pct=True)

for col in d.select_dtypes(include=np.number).columns:
    median_val = d[col].median()
    d[col] = d[col].replace([np.inf, -np.inf], median_val)

    # d[col] = d[col].fillna(median_val)

for col in d.select_dtypes(include=np.number).columns:
    median_val = d[col].median()
    d[col] = d[col].replace([np.inf, -np.inf], median_val)

    # d[col] = d[col].fillna(median_val)


table = pa.Table.from_pandas(d)
pq.write_table(table, './data/soccer_model.parquet',compression='BROTLI')

