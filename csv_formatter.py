
def csv_formatter(matches):
    from collections import defaultdict
    winning_streak = defaultdict(list)

    rows = []
    for match in matches:
        # make the data for the home team
        away_team_average_era = match.away_team_standing['era']
        away_team_starting_pitcher_era = match.pitcher_info['away'][0]['era']

        home_team_batters_hra = list(map(lambda x: x['hra'], sorted(match.batter_info['home'], key=lambda x: x['at_bats'], reverse=True)[:9]))
        is_home = 1
        is_winning_stream = (1 if winning_streak[match.home_team_name].count(1) == 3 else 0)
        if len(winning_streak[match.home_team_name]) == 3:
            winning_streak[match.home_team_name].pop(0)
        home_team_score = sum(match.score_board['scores']['home'])
        away_team_score = sum(match.score_board['scores']['away'])
        winning_streak[match.home_team_name].append(home_team_score >= away_team_score)

        rows.append(','.join(map(str, [home_team_score] + home_team_batters_hra + [away_team_starting_pitcher_era, away_team_average_era, is_home, is_winning_stream])))

        # make the data for the away team.
        home_team_average_era = match.home_team_standing['era']
        home_team_starting_pitcher_era = match.pitcher_info['home'][0]['era']

        away_team_batters_hra = list(map(lambda x: x['hra'], sorted(match.batter_info['away'], key=lambda x: x['at_bats'], reverse=True)[:9]))
        is_home = 0
        is_winning_stream = (1 if winning_streak[match.away_team_name].count(1) == 3 else 0)
        if len(winning_streak[match.away_team_name]) == 3:
            winning_streak[match.away_team_name].pop(0)
        winning_streak[match.away_team_name].append(home_team_score < away_team_score)

        rows.append(','.join(map(str, [away_team_score] + away_team_batters_hra + [home_team_starting_pitcher_era, home_team_average_era, is_home, is_winning_stream])))
    return rows
