import os
import pathlib

class DataFetch(object):

    def __init__(self):

        self.DOWNLOAD_ROOT = "https://raw.githubusercontent.com/robretoarenal/DSF_FinalProject/main/"
        self.DS_PATH = os.path.join("datasets")
        self.GAMES_URL = DOWNLOAD_ROOT + "Data/spreadspoke_scores.csv"
        self.TEAMS_URL = DOWNLOAD_ROOT + "Data/nfl_teams.csv"

    def fetch_data(games_url="Data/spreadspoke_scores.csv", teams_url=TEAMS_URL, ds_path=DS_PATH):
        if not os.path.isdir(ds_path):
            os.makedirs(ds_path)
        games_path = os.path.join(ds_path,'spreadspoke_scores.csv')
        urllib.request.urlretrieve(games_url, games_path)
        teams_path = os.path.join(ds_path,'nfl_teams.csv')
        urllib.request.urlretrieve(teams_url, teams_path)

        return ds_path


class ScoresETL(object):
    def __init__(self, test_size, random_state):

        self.ds_path = DataFetch().fetch_data()
        self.test_size = float(test_size)
        self.random_state = int(random_state)
        self.GAMES_ELO_URL = 'https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv'
        self.GAMES_ELO_LATEST_URL = 'https://projects.fivethirtyeight.com/nfl-api/nfl_elo_latest.csv'

    def _load_games_data(self):
        csv_path = os.path.join(self.ds_path, "spreadspoke_scores.csv")

        return pd.read_csv(csv_path)

    def _load_teams_data(ds_path=DS_PATH):
        csv_path = os.path.join(ds_path, "nfl_teams.csv")

        return pd.read_csv(csv_path)

    def _load_elo_data(games_elo_url=GAMES_ELO_URL):

        return pd.read_csv(games_elo_url)



    def _data_cleaning(self):
        df=_load_games_data()
        teams=_load_teams_data()
        games_elo=_load_elo_data()

        df = df.replace(r'^\s*$', np.nan, regex=True)
        df = df[df['score_home'].notna()]
        df = df[df['team_favorite_id'].notna()]
        df = df[df['over_under_line'].notna()]
        #Change data types
        df['over_under_line'] = df.over_under_line.astype(float)
        df['stadium_neutral'] = df.stadium_neutral.astype(int)
        df['schedule_playoff'] = df.schedule_playoff.astype(int)
        # change data type of date columns
        df['schedule_date'] = pd.to_datetime(df['schedule_date'])
        games_elo['date'] = pd.to_datetime(games_elo['date'])
        #Change formats to date columns
        df['schedule_date'] = df['schedule_date'].dt.strftime('%m/%d/%Y')
        games_elo['date'] = games_elo['date'].dt.strftime('%m/%d/%Y')
        #Mapping columns
        df['team_home'] = df.team_home.map(teams.set_index('team_name')['team_id'].to_dict())
        df['team_away'] = df.team_away.map(teams.set_index('team_name')['team_id'].to_dict())
        #reset index
        df.reset_index(drop=True, inplace=True)
        # creating home favorite and away favorite columns (fill na with 0's)
        df.loc[df.team_favorite_id == df.team_home, 'home_favorite'] = 1
        df.loc[df.team_favorite_id == df.team_away, 'away_favorite'] = 1
        df.home_favorite.fillna(0, inplace=True)
        df.away_favorite.fillna(0, inplace=True)
        # creating over/under (fill na with 0's)
        df.loc[((df.score_home + df.score_away) > df.over_under_line), 'over'] = 1
        df.over.fillna(0, inplace=True)
        # Working schedule_week to be just numbers
        # fixing some schedule_week column errors and converting column to integer data type
        df.loc[(df.schedule_week == '18'), 'schedule_week'] = '17'
        df.loc[(df.schedule_week == 'Wildcard') | (df.schedule_week == 'WildCard'), 'schedule_week'] = '18'
        df.loc[(df.schedule_week == 'Division'), 'schedule_week'] = '19'
        df.loc[(df.schedule_week == 'Conference'), 'schedule_week'] = '20'
        df.loc[(df.schedule_week == 'Superbowl') | (df.schedule_week == 'SuperBowl'), 'schedule_week'] = '21'
        df['schedule_week'] = df.schedule_week.astype(int)
        #drop columns that we dont need for analysis.
        df.drop(['schedule_playoff', 'weather_detail'], axis=1, inplace=True)
        #Change WSH team to WAS, to standarize teams. So then we could join.
        games_elo.loc[games_elo.team1 == 'WSH', 'team1'] = 'WAS'
        games_elo.loc[games_elo.team2 == 'WSH', 'team2'] = 'WAS'

class NflPredict(object):

    def __init__(self):
        self.GAMES_ELO_LATEST_URL = 'https://projects.fivethirtyeight.com/nfl-api/nfl_elo_latest.csv'

    def _load_elo_latest_data(games_elo_latest_url=GAMES_ELO_LATEST_URL):
        return pd.read_csv(games_elo_latest_url)

    def predict(self):
        df=_load_elo_latest_data()
        df=df.loc[df['elo1_post'].isnull() & df['elo1_pre'].notnull(),['team1','team2']]
        return df
