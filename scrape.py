from bs4 import BeautifulSoup
import json
import re
import requests

from constants import TEAM_NAMES, STADIUM_NAMES


class DetailDataNotFoundException(Exception):
    """ An exception thrown when the detailed page does not seem to have the data script tag"""


class MatchSummaryParser(object):
    """A class used to scrape match summary data.

    Supports the following three formats of data fetching.
    1) Get a whole month's match data.
    2) Get a specific day's match data. This returns data for all the matches on the given data.
    """
    URL = 'http://sports.news.naver.com/schedule/index.nhn'

    def __init__(self, year, month, day=None):
        self.year = year
        self.month = month
        self.day = day

    def _get_raw_page(self):
        """ Returns the raw data on some target page. The NaverSports page is used to
        get the raw data.

        url: http://sports.news.naver.com/schedule/index.nhn
        params: category=kbo&year=%s&month=%s
        """
        payload = {
            'category': 'kbo',
            'year': self.year,
            'month': self.month
        }

        return requests.get(self.URL, params=payload).text

    def _extract(self, days):
        """Given days data, extracts match results and turns them into an internal representation
        of a match information, which is the MatchInfo class.

        Defines some inner helper functions.
        """

        def is_no_match(day):
            """ Returns True if day has no game. i.e. Mondays. """
            return (
                len(day.tbody.find_all("tr")) == 1 and
                len(day.tbody.tr.find_all("td")) <= 3
            )

        def get_team_names(match):
            """ Returns a tuple of team names. The first is the away team name and
            the second is the home team name.
            """
            return (
                (
                    TEAM_NAMES.get(match.find("span", class_="team_lft").string),
                    TEAM_NAMES.get(match.find("span", class_="team_rgt").string)
                )
            )

        def get_date(day):
            """ Returns the date of a day. """
            return (
                re.findall(
                    "\d+.\d+",
                    str(day.find_all("span", class_="td_date")[0])
                )[0]
            )

        def is_rain_canceled(match):
            """ Returns True if a match is rain canceled. """
            return "colspan" in str(match)

        def get_scores(match):
            """ Returns a tuple of scores where the first is the away team's score
            and the second is the home team's score. """
            return re.findall("\d+", str(match.find("strong", class_="td_score")))

        def get_hours(match):
            """ Extracts the hour of a match. """
            return (
                re.findall("\d+:\d+", str(match.find("span", class_="td_hour")))[0]
            )

        def get_stadium(match):
            """ Extracts the location of a match. """
            return (
                STADIUM_NAMES.get(match.find_all("span", class_="td_stadium")[1].string)
            )

        def has_yet_to_happen(match):
            """ Returns True if the game is scheduled in the future. """
            return len(get_scores(match)) == 0

        def process_each_day(day):
            """ Given a day, processes all the matches on the day and
            return a list of MatchResult instances.
            """
            if is_no_match(day):
                return []
            date = get_date(day)
            day_result = []

            for each_match in day.find_all("tr"):
                if is_rain_canceled(each_match) or has_yet_to_happen(each_match):
                    continue

                time = get_hours(each_match)
                match_teams = get_team_names(each_match)
                stadium = get_stadium(each_match)
                scores = get_scores(each_match)

                # This is most likely some event game. i.e.
                if None in match_teams:
                    continue

                # Specific teams are specified and this match is not relevant.
                # if teams and (match_teams[0] not in teams and match_teams[1] not in teams):
                #     continue

                day_result.append(
                    MatchSummary(
                        self.year,
                        date,
                        time,
                        match_teams[1],
                        match_teams[0],
                        scores[1],
                        scores[0],
                        stadium
                    )
                )
            return day_result

        result = []
        for each_day in days:
            result += process_each_day(each_day)
        return result

    def parse(self, filter_function=None):
        """ Parses and returns the raw HTML page into some format of choice.
        BeautifulSoup is used to turn the HTML page into some tree format that is easy to traverse.
        Given the tree, extracts the HTML element that contains information we need
        and returns all the div elements inside some table that each corresponds to a single day.

        Note that this code assumes a certain HTML page structure and is thus fragile.

        If some filter_function is given, it is applied and returns some subset of days.
        The filter function must take as input the div element that corresponds to a single day.
        """
        tree = BeautifulSoup(self._get_raw_page(), "html.parser")
        days_table = tree.find_all("div", class_="tb_wrap", id="calendarWrap")[0]
        all_days = days_table.find_all("div", recursive=False)

        # If day is specified, filter by the given day.
        if self.day:
            all_days = list(filter(
                lambda day: (
                    int(re.findall(
                        "\d+.\d+",
                        str(day.find_all("span", class_="td_date")[0])
                    )[0].split(".")[1]) == int(self.day)
                ),
                all_days
            ))

        if filter_function:
            return list(filter(filter_function, all_days))
        return self._extract(all_days)


class MatchSummary(object):
    """ A class that internally represents each match's summary. """
    def __init__(
        self,
        year,
        date,
        time,
        home_team_name,
        away_team_name,
        home_team_score,
        away_team_score,
        stadium
    ):
        self.year = year
        self.month, self.day = date.split('.')
        self.day = self.day.zfill(2)
        self.month = self.month.zfill(2)
        self.time = time
        self.home_team_name = home_team_name
        self.away_team_name = away_team_name
        self.home_team_score = int(home_team_score)
        self.away_team_score = int(away_team_score)
        self.stadium = stadium
        self.winner = (
            self.home_team_name
            if self.home_team_score > self.away_team_score
            else self.away_team_score
        )

    def __str__(self):
        return "{0} {1} {2} {3} {4} vs {5} - {6} : {7}".format(
            self.year,
            self.month,
            self.day,
            self.stadium,
            self.home_team_name,
            self.away_team_name,
            self.home_team_score,
            self.away_team_score
        )

    def __repr__(self):
        return "{0} {1} {2} {3} {4} vs {5} - {6} : {7}".format(
            self.year,
            self.month,
            self.day,
            self.stadium,
            self.home_team_name,
            self.away_team_name,
            self.home_team_score,
            self.away_team_score
        )

    def get_home_team_name(self):
        return self.home_team_name

    def get_home_team_score(self):
        return self.home_team_score

    def get_away_team_name(self):
        return self.away_team_name

    def get_away_team_score(self):
        return self.away_team_score

    def get_winner(self):
        return self.winner

    def to_json(self):
        return json.dumps(self.__dict__)


class MatchDetailParser(object):
    """ A class that scrapes the details of a specific match between the given two teams
    on a given date and turns it into an internal representation of a match detail.

    This data includes
    0) The key events of a game.
    1) The scoreboard of each inning
    2) The batter breakdown result of how each batter performed for the match.
    This includes the batter's batting average as of the end of that game and how many
    hits (안타), runs (득점. 홈을 밟은 횟수), RBI (타점. 홈으로 불러들인 횟수) he made.
    3) The pitcher breakdown result of how each pitcher performed.
    This includes the pitcher's total # of innings, how many batters he took on,
    how many hits, four balls, home runs he allowed, how many strikeouts he scored
    how many scores he allowed, how many mistakes he was at fault for and his ERA
    by the end of the game.
    """
    # The mapping between the internal team names and the scraper specific names.
    # The team names not in the mapping are identical.
    TEAM_NAME_MAPPING = {
        'HANHWA': 'HH',
        'KT': 'KT',
        'KIA': 'HT',
        'NEXEN': 'WO',
        'DOOSAN': 'OB',
        'SAMSUNG': 'SS',
        'LOTTE': 'LT'
    }

    # The mapping between the scrapper specific names and hte internal team names.
    REVERSE_NAME_MAPPING = {v: k for k, v in TEAM_NAME_MAPPING.items()}

    URL = 'http://sports.news.naver.com/gameCenter/gameRecord.nhn'

    def __init__(self, year, month, day, away_team_name, home_team_name):
        """home_team_name and away team name must be in the format specified in constants.py
        More concretely, it has to be one of the values of TEAM NAMES.
        """
        self.year = year
        self.month = month
        self.day = day
        self.home_team_name = self.TEAM_NAME_MAPPING.get(home_team_name, home_team_name)
        self.away_team_name = self.TEAM_NAME_MAPPING.get(away_team_name, away_team_name)
        self.game_id_factory = GameIDFactory(
            self.year,
            self.month,
            self.day,
            self.away_team_name,
            self.home_team_name,
        )

    def _get_raw_page(self, game_id):
        """ Returns the raw data on some target page. The NaverSports game result page is used to
        get the raw data.

        url: http://sports.news.naver.com/gameCenter/gameResult.nhn
        params: category=kbo&gameId=%s
        """
        payload = {
            'category': 'kbo',
            'gameId': game_id
        }
        return requests.get(self.URL, params=payload).text

    def _parse_source_script_that_has_data(self):
        """ Parses the raw string of a script tag so that we can extract out the data part.
        This is unfortunately done because Naver renders the actual data
        using Javascript on the client side.

        Very ugly piece of code. Basically, finds the source tag that has the actual data
        using some keyword and then extracts out the json formatted data by some
        custom string processing.
        """
        tree = BeautifulSoup(self._get_raw_page(self.game_id_factory.make()), "html.parser")
        scripts = tree.find_all("script")

        # Note that the logic sadly relies on these two magic keywords positions.
        magic_keyword = 'DataClass = jindo.$Class('
        magic_keyword_two = '_data'

        data_script = None
        for script in scripts:
            if magic_keyword in str(script):
                data_script = str(script)
                break

        # This will most likely be a double header game.
        if data_script is None:
            raise DetailDataNotFoundException()

        # Process this string by looking for some valid JSON format
        argument_part = data_script[data_script.index(magic_keyword) + len(magic_keyword):]
        data_part = argument_part[argument_part.index(magic_keyword_two) + len(magic_keyword_two):]

        string_of_interest = data_part[data_part.index('{'):]
        paren_count = 1

        # Algorithm 101. Finds the end position of json data by keeping track of
        # the numbers of {, }.
        idx = 1
        while paren_count > 0:
            if string_of_interest[idx] == '{':
                paren_count += 1
            elif string_of_interest[idx] == '}':
                paren_count -= 1
            idx += 1
        return json.loads(string_of_interest[:idx])

    def parse(self):
        """ Parses the raw page for the following information.
        1) pitcher breakdown information 2) batter breakdown information
        3) per inning scoreboard. 4) away team standing 5) home team standing.
        """
        data_in_json = self._parse_source_script_that_has_data()

        away_team_standing = {
            'draws': int(data_in_json['awayStandings']['d']),
            'era': float(data_in_json['awayStandings']['era']),
            'hra': float(data_in_json['awayStandings']['hra']),
            'wra':  float(data_in_json['awayStandings']['wra']),
            'wins': int(data_in_json['awayStandings']['w']),
            'loses': int(data_in_json['awayStandings']['l']),
            'rank': int(data_in_json['awayStandings']['rank']),
            'name': TEAM_NAMES[data_in_json['awayStandings']['name']],
        }

        home_team_standing = {
            'draws': int(data_in_json['homeStandings']['d']),
            'era': float(data_in_json['homeStandings']['era']),
            'hra': float(data_in_json['homeStandings']['hra']),
            'wra': float(data_in_json['homeStandings']['wra']),
            'wins': int(data_in_json['homeStandings']['w']),
            'loses': int(data_in_json['homeStandings']['l']),
            'rank': int(data_in_json['homeStandings']['rank']),
            'name': TEAM_NAMES[data_in_json['homeStandings']['name']],
        }

        # R = 스코어, H = 안타, E = 실수,에러, B = 볼넷 혹은 몸에 맞는 공.
        score_board = {
            'scores': data_in_json['scoreBoard']['inn'],
            'summary': data_in_json['scoreBoard']['rheb']
        }

        pitcher_info = {
            'home': [],
            'away': []
        }

        batter_info = {
            'home': [],
            'away': []
        }

        for side in ['home', 'away']:
            for pitcher in data_in_json['pitchersBoxscore'][side]:
                pitcher_info[side].append(
                    {
                        'at_bats': int(pitcher['ab']),  # 타수
                        'hits': int(pitcher['hit']),  # 안타 맞은 수
                        'bbhp': int(pitcher['bbhp']),  # 4사
                        'home_runs': int(pitcher['hr']),  # 홈런 맞은 수
                        'strike_outs': int(pitcher['kk']),  # 스트라이크 잡은 수
                        'scores_lost': int(pitcher['r']),  # 내준 점수
                        'errors': int(pitcher['er']),  # 본인 실수
                        'era': float(pitcher['era']),  # 게임 종료 시점의 방어율
                        'name': pitcher['name'],
                        # TODO: process the 1/2, 2/3 unicode and don't round down.
                        'innings': int(pitcher['inn'][0]),  # 던진 이닝 수. 내림
                        'wins': int(pitcher['w']),  # 투수 승수
                        'loses': int(pitcher['l']),  # 투수 패수
                        'saves': int(pitcher['s']),  # 투수 세이브수
                        'num_balls_thrown': int(pitcher['bf']),  # 던진 공 수
                        'game_count': int(pitcher['gameCount']),  # 총 게임 참여 수
                    }
                )

        for side in ['home', 'away']:
            for batter in data_in_json['battersBoxscore'][side]:
                batter_info[side].append(
                    {
                        'at_bats': int(batter['ab']),  # 타석 참여 횟수
                        'hits': int(batter['hit']),  # 안타 수
                        'hra': float(batter['hra']),  # 게임 종료 시점의 타율
                        'rbi': int(batter['rbi']),  # 타점
                        'runs': int(batter['run']),  # 득점
                        'name': batter['name']
                    }
                )

        return MatchDetail(
            self.year,
            self.month,
            self.day,
            self.REVERSE_NAME_MAPPING.get(self.away_team_name, self.away_team_name),
            self.REVERSE_NAME_MAPPING.get(self.home_team_name, self.home_team_name),
            score_board,
            pitcher_info,
            batter_info,
            away_team_standing,
            home_team_standing
        )


class MatchDetail(object):
    """ A data structure that is an internal representation of a match's details.
    Includes 1) pitcher breakdown information 2) batter breakdown information
    3) per inning scoreboard. 4) away team standing 5) home team standing.
    """
    def __init__(
        self,
        year,
        month,
        day,
        away_team_name,
        home_team_name,
        score_board,
        pitcher_info,
        batter_info,
        away_team_standing,
        home_team_standing
    ):
        self.year = year
        self.month = month
        self.day = day
        self.away_team_name = away_team_name
        self.home_team_name = home_team_name
        self.score_board = score_board
        self.pitcher_info = pitcher_info
        self.batter_info = batter_info
        self.away_team_standing = away_team_standing
        self.home_team_standing = home_team_standing

    def to_json(self):
        return json.dumps(self.__dict__)


class GameIDFactory(object):
    """Given year, month, day, home_team_name, away_team_name, constructs the corresponding
    gameID used to go to the Naver Sports page.
    """
    def __init__(
        self,
        year,
        month,
        day,
        away_team_name,
        home_team_name
    ):
        self.year = year
        self.month = month
        self.day = day
        self.away_team_name = away_team_name
        self.home_team_name = home_team_name

    def make(self):
        if int(self.year) >= 2016:
            return '{0}{1}{2}{3}{4}0{0}'.format(
                self.year,
                self.month,
                self.day,
                self.away_team_name,
                self.home_team_name,
            )
        else:
            return '{0}{1}{2}{3}{4}0'.format(
                self.year,
                self.month,
                self.day,
                self.away_team_name,
                self.home_team_name,
            )
