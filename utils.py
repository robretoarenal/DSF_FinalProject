import pandas as pd
import numpy as np
import datetime
import sklearn
import pickle
import os
import pathlib
import urllib.request
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from pickle import dump, load

class DataFetch(object):

    def __init__(self):
        self.DOWNLOAD_ROOT = "https://raw.githubusercontent.com/robretoarenal/DSF_FinalProject/main/"
        self.DS_PATH = os.path.join("datasets")
        #self.GAMES_URL = DOWNLOAD_ROOT + "Data/spreadspoke_scores.csv"
        self.TEAMS_URL = self.DOWNLOAD_ROOT + "datasets/nfl_teams.csv"

    def fetch_data(self):
        if not os.path.isdir(self.DS_PATH):
            os.makedirs(self.DS_PATH)
        #games_path = os.path.join(ds_path,'spreadspoke_scores.csv')
        #urllib.request.urlretrieve(games_url, games_path)
        teams_path = os.path.join(self.DS_PATH,'nfl_teams.csv')
        urllib.request.urlretrieve(self.TEAMS_URL, teams_path)

        return self.DS_PATH

class GenerateAttributes(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ready=1

    def _prepareAttributes(self, X):

        #Create new table with one row per team per game.
        games = X.groupby(['season','date', 'team1']).mean()[['score1', 'score2']].reset_index()
        #games = X.groupby(['season','date', 'team1']).agg({'score1':'mean','score2':'mean'}).reset_index()
        #games = X.groupby(['season','date', 'team1'])[['score1','score2']].mean().reset_index()
        aw_games = X.groupby(['season','date', 'team2']).mean()[['score1', 'score2']].reset_index()
        #aw_games = X.groupby(['season','date', 'team2']).agg({'score1':'mean','score2':'mean'}).reset_index()
        #aw_games = X.groupby(['season','date', 'team2'])[['score1','score2']].mean().reset_index()
        games['point_diff'] = games.score1 - games.score2
        aw_games['point_diff'] = aw_games.score2 - aw_games.score1
        # append the two dataframes
        games = games.append(aw_games, ignore_index=True, sort=True)
        # fill null values
        games.team1.fillna(games.team2, inplace=True)
        # sort by season and week
        games.sort_values(['date'], ascending = [True], inplace=True)
        # removing unneeded columns & changing column name
        games = games[['season','date', 'team1', 'score1', 'score2', 'point_diff']]
        games.rename(columns={'team1' : 'team'}, inplace=True)
        tm_dict = {}
        for key in games.team.unique():
            tm_dict[key] = games[games.team == key].reset_index(drop=True)

        pts_diff = pd.DataFrame()

        for yr in games.season.unique():
          #print(yr)
          for tm in games.team.unique():
            #print(tm)
            data = tm_dict[tm].copy()
            data = data[data.season == yr]
            data.loc[:, 'avg_pts_diff'] = data.point_diff.shift().expanding().mean()
            data.loc[:, 'total_pts'] = data.score1.shift().expanding().mean()
            data.loc[:, 'total_against_pts'] = data.score2.shift().expanding().mean()
            pts_diff = pts_diff.append(data)

        #Add week column just for first week of season(NaNs)
        pts_diff.loc[(pts_diff.avg_pts_diff.isna()),'Week'] = 1
        pts_diff.Week.fillna(0, inplace=True)
        pts_diff['Week'] = pts_diff['Week'].astype(int)

        #Create 3 different dataFrames to group by each stat.
        total_season = pts_diff.groupby(['season','team']).mean()['score1'].reset_index()
        #total_season = pts_diff.groupby(['season','date']).agg({'score1':'mean'}).reset_index()
        #total_season = pts_diff.groupby(['season','date'])[['score1']].mean().reset_index()
        total_season2= pts_diff.groupby(['season','team']).mean()['score2'].reset_index()
        #total_season2 = pts_diff.groupby(['season','date']).agg({'score2':'mean'}).reset_index()
        #total_season2 = pts_diff.groupby(['season','date'])[['score2']].mean().reset_index()
        total_season3 = pts_diff.groupby(['season','team']).mean()['point_diff'].reset_index()
        #total_season3 = pts_diff.groupby(['season','date']).agg({'point_diff':'mean'}).reset_index()
        #total_season3 = pts_diff.groupby(['season','date'])[['point_diff']].mean().reset_index()

        #Merge the 3 different grouped by stats (score1, score2, point_diff) and rename them (total_pts, total_against_pts, avg_pts_diff)
        total_season = total_season.merge(total_season2[['season','team','score2']],
                                          left_on=['season','team'], right_on=['season','team'], how='left')

        total_season = total_season.merge(total_season3[['season','team','point_diff']],
                                          left_on=['season','team'], right_on=['season','team'], how='left')

        #total_season = total_season.merge(weeks1[['season','team','date']],left_on=['season','team'],right_on=['season','team'], how='left')

        total_season.rename(columns={'score1' : 'total_pts'}, inplace=True)
        total_season.rename(columns={'score2' : 'total_against_pts'}, inplace=True)
        total_season.rename(columns={'point_diff' : 'avg_pts_diff'}, inplace=True)

        #Add Week = 1 column and pass to next season so it be the weeks 1 stat
        total_season['Week'] = 1
        total_season['season'] += 1

        #Merge
        pts_diff = pts_diff.merge(total_season[['season','team','Week','total_pts','total_against_pts','avg_pts_diff']],
                                 left_on=['season','team','Week'],
                                  right_on=['season','team','Week'], how='left')

        pts_diff.avg_pts_diff_x.fillna(pts_diff.avg_pts_diff_y, inplace=True)
        pts_diff.total_pts_x.fillna(pts_diff.total_pts_y, inplace=True)
        pts_diff.total_against_pts_x.fillna(pts_diff.total_against_pts_y, inplace=True)
        #drop _y columns and rename _x columns
        pts_diff.drop(columns=['avg_pts_diff_y','total_pts_y','total_against_pts_y'], inplace=True)
        pts_diff.columns = pts_diff.columns.str.replace('_x', '')

        return pts_diff

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pts_diff = self._prepareAttributes(X)
        #merge new attributes to the original dataframe. For both home and away teams.
        X = X.merge(pts_diff[['season','date','team','avg_pts_diff','total_pts','total_against_pts']],
              left_on=['season','date','team1'],
              right_on=['season','date','team'], how='left')
        X.rename(columns={'avg_pts_diff' : 'hm_avg_diff'}, inplace=True)
        X.rename(columns={'total_pts' : 'hm_avg_pts'}, inplace=True)
        X.rename(columns={'total_against_pts' : 'hm_avg_against_pts'}, inplace=True)

        X = X.merge(pts_diff[['season','date','team','avg_pts_diff','total_pts','total_against_pts']],
              left_on=['season','date','team2'],
              right_on=['season','date','team'], how='left')
        X.rename(columns={'avg_pts_diff' : 'aw_avg_diff'}, inplace=True)
        X.rename(columns={'total_pts' : 'aw_avg_pts'}, inplace=True)
        X.rename(columns={'total_against_pts' : 'aw_avg_against_pts'}, inplace=True)

        #return the table ready
        return X

class newClass(object):
    def __init__(self):
        self.ready=1

    def _prepareAttributes(self, df):
        #Create new table with one row per team per game.
        games = df.groupby(['season','date', 'team1']).mean()[['score1', 'score2']].reset_index()
        aw_games = df.groupby(['season','date', 'team2']).mean()[['score1', 'score2']].reset_index()
        games['point_diff'] = games.score1 - games.score2
        aw_games['point_diff'] = aw_games.score2 - aw_games.score1
        # append the two dataframes
        games = games.append(aw_games, ignore_index=True, sort=True)

        # fill null values
        games.team1.fillna(games.team2, inplace=True)

        # sort by season and week
        games.sort_values(['date'], ascending = [True], inplace=True)

        # removing unneeded columns & changing column name
        games = games[['season','date', 'team1', 'score1', 'score2', 'point_diff']]
        games.rename(columns={'team1' : 'team'}, inplace=True)
        tm_dict = {}
        for key in games.team.unique():
            tm_dict[key] = games[games.team == key].reset_index(drop=True)

        pts_diff = pd.DataFrame()

        for yr in games.season.unique():
          #print(yr)
          for tm in games.team.unique():
            #print(tm)
            data = tm_dict[tm].copy()
            data = data[data.season == yr]
            data.loc[:, 'avg_pts_diff'] = data.point_diff.shift().expanding().mean()
            data.loc[:, 'total_pts'] = data.score1.shift().expanding().mean()
            data.loc[:, 'total_against_pts'] = data.score2.shift().expanding().mean()
            pts_diff = pts_diff.append(data)

        #Add week column just for first week of season(NaNs)
        pts_diff.loc[(pts_diff.avg_pts_diff.isna()),'Week'] = 1
        pts_diff.Week.fillna(0, inplace=True)
        pts_diff['Week'] = pts_diff['Week'].astype(int)

        #Create 3 different dataFrames to group by each stat.
        total_season = pts_diff.groupby(['season','team']).mean()['score1'].reset_index()
        total_season2= pts_diff.groupby(['season','team']).mean()['score2'].reset_index()
        total_season3 = pts_diff.groupby(['season','team']).mean()['point_diff'].reset_index()

        #Merge the 3 different grouped by stats (score1, score2, point_diff) and rename them (total_pts, total_against_pts, avg_pts_diff)
        total_season = total_season.merge(total_season2[['season','team','score2']],
                                          left_on=['season','team'], right_on=['season','team'], how='left')

        total_season = total_season.merge(total_season3[['season','team','point_diff']],
                                          left_on=['season','team'], right_on=['season','team'], how='left')


        total_season.rename(columns={'score1' : 'total_pts'}, inplace=True)
        total_season.rename(columns={'score2' : 'total_against_pts'}, inplace=True)
        total_season.rename(columns={'point_diff' : 'avg_pts_diff'}, inplace=True)

        #Add Week = 1 column and pass to next season so it be the weeks 1 stat
        total_season['Week'] = 1
        total_season['season'] += 1

        #Merge
        pts_diff = pts_diff.merge(total_season[['season','team','Week','total_pts','total_against_pts','avg_pts_diff']],
                                 left_on=['season','team','Week'],
                                  right_on=['season','team','Week'], how='left')

        pts_diff.avg_pts_diff_x.fillna(pts_diff.avg_pts_diff_y, inplace=True)
        pts_diff.total_pts_x.fillna(pts_diff.total_pts_y, inplace=True)
        pts_diff.total_against_pts_x.fillna(pts_diff.total_against_pts_y, inplace=True)
        #drop _y columns and rename _x columns
        pts_diff.drop(columns=['avg_pts_diff_y','total_pts_y','total_against_pts_y'], inplace=True)
        pts_diff.columns = pts_diff.columns.str.replace('_x', '')

        return pts_diff

    def mergeAttributes(self, df):

        pts_diff=self._prepareAttributes(df)
        df = df.merge(pts_diff[['season','date','team','avg_pts_diff','total_pts','total_against_pts']],
              left_on=['season','date','team1'],
              right_on=['season','date','team'], how='left')
        df.rename(columns={'avg_pts_diff' : 'hm_avg_diff'}, inplace=True)
        df.rename(columns={'total_pts' : 'hm_avg_pts'}, inplace=True)
        df.rename(columns={'total_against_pts' : 'hm_avg_against_pts'}, inplace=True)

        df = df.merge(pts_diff[['season','date','team','avg_pts_diff','total_pts','total_against_pts']],
              left_on=['season','date','team2'],
              right_on=['season','date','team'], how='left')
        df.rename(columns={'avg_pts_diff' : 'aw_avg_diff'}, inplace=True)
        df.rename(columns={'total_pts' : 'aw_avg_pts'}, inplace=True)
        df.rename(columns={'total_against_pts' : 'aw_avg_against_pts'}, inplace=True)

        #return the table ready
        return df


class ScoresETL(object):
    def __init__(self, random_state):

        #self.ds_path = DataFetch().fetch_data()
        self.random_state = int(random_state)
        self.GAMES_ELO_URL = 'https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv'
        #self.GAMES_ELO_LATEST_URL = 'https://projects.fivethirtyeight.com/nfl-api/nfl_elo_latest.csv'

    def _load_elo_data(self):

        return pd.read_csv(self.GAMES_ELO_URL)

    def _data_cleaning(self):
        df = self._load_elo_data()

        #change date to datetime
        df['date'] = pd.to_datetime(df['date'])
        #add new column season and adjust season of january and february
        df['season'] = pd.DatetimeIndex(df['date']).year
        df.loc[df['date'].dt.month < 3, 'season'] = df['season'] - 1
        #add column of result
        df['result'] = (df.score1 >= df.score2).astype(int)
        #Remove future games rows
        df = df[df['team1'].notna()]
        #Filter rows that we need onlny
        #df = df[['season','date','playoff','team1','team2','score1','score2','qbelo1_pre','qbelo2_pre','qbelo_prob1','qb1_game_value','qb2_game_value','result']]
        df = df[['season','date','playoff','team1','team2','score1','score2','qbelo1_pre','qbelo2_pre','qbelo_prob1','qb1_value_pre','qb2_value_pre','result']]

        return df

    def etl_pipeline(self):
        df = self._data_cleaning()
        train_set = df.loc[df['season'] < 2017]
        test_set = df.loc[df['season'] > 2016]

        pipeline = Pipeline([
            ('attribs_adder', GenerateAttributes())
        ])

        train_set_full = pipeline.fit_transform(train_set)
        train_set_full = train_set_full.dropna(how='any',axis=0)
        #features_full = train_set_full[['qbelo1_pre','qbelo2_pre','qbelo_prob1','qb1_game_value','qb2_game_value','hm_avg_diff','aw_avg_diff']]
        features_full = train_set_full[['qbelo1_pre','qbelo2_pre','qbelo_prob1','qb1_value_pre','qb2_value_pre','hm_avg_diff','aw_avg_diff']]
        results_full = train_set_full["result"].copy()

        scaler_path = os.path.join(pathlib.Path().absolute(), "scaler")
        scaler_file = scaler_path + "/scaled_features.pkl"

        if not os.path.isdir(scaler_path):
            os.makedirs(scaler_path)
        dump(train_set_full, open(scaler_file, 'wb'))

        return features_full, results_full

class ScoresTrain(object):

    def __init__(self,results_full, features_full, n_estimators):
        self.results_full = results_full
        self.features_full = features_full
        self.n_estimators = int(n_estimators)

    def train(self):

        forest_reg = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
        forest_reg.fit(self.features_full, self.results_full)

        # Save the model
        model_path = os.path.join(str(pathlib.Path().absolute()), "model")
        model_file = model_path + "/forest_reg.pkl"

        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        dump(forest_reg, open(model_file, 'wb'))


class ScoresPredict(object):

    def __init__(self):
        self.GAMES_ELO_URL = 'https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv'
        scaler_path = os.path.join(str(pathlib.Path().absolute()), "scaler")
        scaler_file = scaler_path + "/scaled_features.pkl"
        model_path = os.path.join(str(pathlib.Path().absolute()), "model")
        model_file = model_path + "/forest_reg.pkl"

        self.scaler = load(open(scaler_file, 'rb'))
        self.model = load(open(model_file, 'rb'))

    def _load_elo_data(self):
        return pd.read_csv(self.GAMES_ELO_URL)
        #return pd.read_csv("nfl_elo.csv")

    def _load_teams_data(self):
        csv_path = os.path.join(self.ds_path, "nfl_teams.csv")

        return pd.read_csv(csv_path)

    def _prepareData(self, df):
        #change date to datetime
        df['date'] = pd.to_datetime(df['date'])
        #add new column season and adjust season of january and february
        df['season'] = pd.DatetimeIndex(df['date']).year
        df.loc[df['date'].dt.month < 3, 'season'] = df['season'] - 1
        #January and February(superbowl) are considered from the last year season
        now = datetime.datetime.now()
        actualSeason=now.year - 1 if now.month < 3 else now.year
        #We need the last season to get the stats of the first game of the actual season.
        lastSeasons=actualSeason-1
        df = df.loc[(df['season']<=actualSeason) & (df['season']>=lastSeasons)]
        #merge the name of the team column
            #clean the teams catalog
        teams = self._load_teams_data()
        teams = teams[teams['team_division'].notna()]
        teams = teams.loc[(teams['team_name'] != 'San Diego Chargers') & (teams['team_name'] != 'Washington Redskins')]
        teams.loc[teams['team_id']=='LVR' , 'team_id'] = 'OAK'
        teams.loc[teams['team_id']=='WAS' , 'team_id'] = 'WSH'
        df = df.merge(teams[['team_name','team_id']],left_on=['team1'], right_on=['team_id'], how='left')
        df.rename(columns={'team_name' : 'hm_team_name'}, inplace=True)
        df = df.merge(teams[['team_name','team_id']],left_on=['team2'], right_on=['team_id'], how='left')
        df.rename(columns={'team_name' : 'aw_team_name'}, inplace=True)

        return df

    def predict(self):

        self.ds_path = DataFetch().fetch_data()
        df=self._load_elo_data()
        df=self._prepareData(df)
        #df_trans = self.scaler.transform(df)
        nc = newClass()
        df_trans = nc.mergeAttributes(df)
        #exclude games of past and future weeks.
        df_trans = df_trans[df_trans['score1'].isna()]
        df_trans = df_trans[df_trans['hm_avg_diff'].notna()]
        #Choose the features to preduct.
        #df_features = df_trans[['qbelo1_pre','qbelo2_pre','qbelo_prob1','qb1_game_value','qb2_game_value','hm_avg_diff','aw_avg_diff']]
        df_features = df_trans[['elo1_pre','elo2_pre','qbelo_prob1','qb1_value_pre','qb2_value_pre','hm_avg_diff','aw_avg_diff']]
        #predict results.
        results=self.model.predict(df_features)
        #select the columns to show on the screen and add the results columns.
        df_trans = df_trans[['season','date','hm_team_name','aw_team_name']]
        df_trans.rename(columns={'hm_team_name':'Home Team', 'aw_team_name':'Away Team'}, inplace=True)
        df_trans.loc[:,'Home win prob'] = results
        df_trans.loc[:,'Away win prob'] = 1 - results
        df_trans['Home win prob'] = pd.Series(["{0:.0%}".format(val) for val in df_trans['Home win prob']], index = df_trans.index)
        df_trans['Away win prob'] = pd.Series(["{0:.0%}".format(val) for val in df_trans['Away win prob']], index = df_trans.index)

        return df_trans
