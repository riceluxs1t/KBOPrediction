from bs4 import BeautifulSoup
import json
import re
import requests

from constants import TEAM_NAMES, STADIUM_NAMES


class DateParsingException(Exception):
    """ An exception thrown when the input parsing date is malformed. """


class DetailDataNotFoundException(Exception):
    """ An exception thrown when the detailed page does not seem to have the data script tag"""


class DayDataParser(object):
    """ A class used to scrape day's data.

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
                    MatchResult(
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


class MatchResult(object):
    """ A class that internally represents each match. """
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
        self.date = self.convert_date_to_num(date)
        self.time = self.convert_date_to_num(time)
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
            self.date,
            self.time,
            self.stadium,
            self.home_team_name,
            self.away_team_name,
            self.home_team_score,
            self.away_team_score
        )

    def __repr__(self):
        return "{0} {1} {2} {3} {4} vs {5} - {6} : {7}".format(
            self.year,
            self.date,
            self.time,
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
        return self.home_team_name

    def get_away_team_score(self):
        return self.home_team_score

    def get_winner(self):
        return self.winner

    def to_json(self):
        return json.dumps(self.__dict__)

    def convert_date_to_num(self, date):
        if '.' in date:
            a = date.split('.')
            if int(a[0]) < 10:
                a[0] = '0' + a[0]
            if int(a[1]) < 10:
                a[1] = '0' + a[1]
            rst = str(self.year) + a[0] + a[1]
            return int(rst)
        elif ':' in date:
            a = date.split(':')
            rst = a[0] + a[1]
            return int(rst)

        raise DateParsingException()


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
    TEAM_NAMES = {
        'SK': 'SK',
        'KIA': 'KIA',
        'NC': 'NC',
        'LG': 'LG',
        'kt': 'KT',
        '삼성': 'SAMSUNG',
        '두산': 'DOOSAN',
        '넥센': 'NEXEN',
        '롯데': 'LOTTE',
        '한화': 'HANHWA',
    }

    # The mapping between the internal team names and the scraper specific names.
    # The team names not in the mapping are identical.
    TEAM_NAME_MAPPING = {
        'HANHWA': 'HH',
        'KT': 'KT',
        'KIA': 'HT',
        'NEXEN': 'WO',
        'DOOSAN': 'OB',
        'SAMSUNG': 'SS',
    }

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
        self.game_id = '{0}{1}{2}{3}{4}0{0}'.format(
            self.year,
            self.month,
            self.day,
            self.away_team_name,
            self.home_team_name,
        )

    def _get_raw_page(self):
        """ Returns the raw data on some target page. The NaverSports game result page is used to
        get the raw data.

        url: http://sports.news.naver.com/gameCenter/gameResult.nhn
        params: category=kbo&gameId=%s
        """
        payload = {
            'category': 'kbo',
            'gameId': self.game_id
        }

        return requests.get(self.URL, params=payload).text

    def _parse_source_script_that_has_data(self, page):
        """ Parses the raw string of a script tag so that we can extract out the data part.
        This is unfortunately done because Naver renders the actual data
        using Javascript on the client side.

        Very ugly piece of code. Basically, finds the source tag that has the actual data
        using some keyword and then extracts out the json formatted data by some
        custom string processing.
        """
        tree = BeautifulSoup(page, "html.parser")
        scripts = tree.find_all("script")

        # Note that the logic sadly relies on these two magic keywords positions.
        magic_keyword = 'DataClass = jindo.$Class('
        magic_keyword_two = '_data'

        data_script = None
        for script in scripts:
            if magic_keyword in str(script):
                data_script = str(script)
                break

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

    def parse(self, page):
        """ Parses the raw page for some information.
        The info_type parameter must be a valid key. i.e. one of
        1) etcRecords: key evetns information
        2) pitchersBoxscore: pitcher breakdown information
        3) battersBoxscore: batter breakdown information
        4) scoreBoard: scoreboard information
        """
        data_in_json = self._parse_source_script_that_has_data(page)
        # TODO: finalize the internal data structure


class MatchDetail(object):
    """ A data structure that is an internal representation of a match's details.
    Includes 1) key events, 2) pitcher breakdown information 3) batter breakdown information
    4) per inning scoreboard.
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
        key_events
    ):
        self.year = year
        self.month = month
        self.day = day
        self.away_team_name = away_team_name
        self.home_team_name = home_team_name
        self.score_board = score_board
        self.pitcher_info = pitcher_info
        self.batter_info = batter_info
        self.key_events = key_events
