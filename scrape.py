from bs4 import BeautifulSoup
import json
import re
import requests

from constants import TEAM_NAMES, STADIUM_NAMES


class DateParsingException(Exception):
    """ An exception thrown when the input parsing date is malformed. """


class DayDataScraper(object):
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

    def get_raw_page(self):
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

    def parse_raw_page(self, page, filter_function=None):
        """ Parses and returns the raw HTML page into some format of choice.
        BeautifulSoup is used to turn the HTML page into some tree format that is easy to traverse.
        Given the tree, extracts the HTML element that contains information we need
        and returns all the div elements inside some table that each corresponds to a single day.

        Note that this code assumes a certain HTML page structure and is thus fragile.

        If some filter_function is given, it is applied and returns some subset of days.
        The filter function must take as input the div element that corresponds to a single day.
        """
        tree = BeautifulSoup(page, "html.parser")
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
        return all_days


class DayInterpreter(object):
    """ An interpreter class that provides some ways of interpreting each "day". """
    def __init__(self, year, month):
        self.year = year
        self.month = month

    def extract(self, days, teams=None):
        """Given days data, extracts match results. Each match result is an instance of
        MatchInfo class. If teams is specified, only returns those teams's matches.
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
                if teams and (match_teams[0] not in teams and match_teams[1] not in teams):
                    continue

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


class MatchResult(object):
    """ A class that represents each match. """
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
