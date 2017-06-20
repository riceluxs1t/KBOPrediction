## Note
Uses python3.6

## Usage
In order to obtain the entire 2017 KBO baseball data, simply run the following.
```python
from scrape import *
year = '2017'
months = ['03', '04', '05', '06']
summaries = []
for month in months:
    summaries += MatchSummaryParser(year, month).parse()

matches = []
for summary in summaries:
    matches.append(
        MatchDetailParser(
            summary.year,
            summary.month,
            summary.day,
            summary.get_away_team_name(),
            summary.get_home_team_name()
        ).parse()
    )
```

## What each match data looks like
```json
{
    "year": "2017",
    "month": "03",
    "day": "14",
    "away_team_name": "KT",
    "home_team_name": "SAMSUNG",
    "score_board": {
      "scores": {
        "away": [
          3,
          1,
          1,
          0,
          1,
          0,
          2,
          0,
          1
        ],
        "home": [
          0,
          0,
          0,
          0,
          1,
          0,
          0,
          0,
          0
        ]
      },
      "summary": {
        "away": {
          "r": 9,  // 점수
          "b": 6,  // 볼넷 + 사사구(몸에 맞는 공)
          "e": 0,  // 팀 실수
          "h": 12  // 안타
        },
        "home": {
          "r": 1,
          "b": 3,
          "e": 0,
          "h": 7
        }
      }
    },
    "pitcher_info": {
      "home": [
        {
          "at_bats": 11,  // 상대한 타자 수
          "hits": 6,  // 맞은 안타 수
          "bbhp": 3,  // 던진 볼넷 + 사사구
          "home_runs": 0,  // 맞은 홈런 수
          "strike_outs": 1,  // 잡은 스트라이크 아웃 수
          "scores_lost": 5,  // 투수가 내준 점수
          "errors": 5,  // 투수 실수
          "era": "15.00",  // 방어율
          "name": "\ucd5c\ucda9\uc5f0",
          "innings": "3",  // 던진 이닝 수. 단, 내림 처리 하였기 때문에 3 1/2, 2/3 모두 3으로 내림.
          "wins": 0,  // 투수의 시즌 승수
          "loses": 1,  // 투수의 시즌 패배 수
          "saves": 0,  // 투수의 시즌 세이브 수
          "num_balls_thrown": 60,  // 해당 경기 던진 공 수
          "game_count": 1  // 투수의 시즌 게임 수
        },
        ...
      ],
      "away": [
        {
          "at_bats": 18,
          "hits": 6,
          "bbhp": 1,
          "home_runs": 0,
          "strike_outs": 1,
          "scores_lost": 1,
          "errors": 1,
          "era": "1.80",
          "name": "\ub85c\uce58",
          "innings": "5",
          "wins": 1,
          "loses": 0,
          "saves": 0,
          "num_balls_thrown": 72,
          "game_count": 1
        },
        ...
      ]
    },
    "batter_info": {
      "home": [
        {
          "at_bats": 2,  // 타자가 해당 경기에 타석에 들어간 횟 수
          "hits": 1,  // 안타
          "hra": "0.500",  // 해당 경기종료 시점의 타율
          "rbi": 1,  // 타점 (때려서 홈으로 불러 들여온 수)
          "runs": 0,  // 득점 (본인이 홈을 밟은 수)
          "name": "\ubc15\ud574\ubbfc"
        },
        ...
      ],
      "away": [
        {
          "at_bats": 3,
          "hits": 1,
          "hra": "0.333",
          "rbi": 0,
          "runs": 1,
          "name": "\uc774\ub300\ud615"
        },
        ...
      ]
    },
    "away_team_standing": {
      "draws": 0,  // 종료 시점의 팀의 무승부 수
      "era": 5.53,  // 종료 시점의 팀의 평균 방어율
      "hra": 0.264,  // 종료 시점의 팀의 평균 타율
      "wra": 0.373,  // 종료 시점의 팀의 승률
      "wins": 25,  // 종료 시점의 팀의 승수
      "loses": 42,  // 종료 시점의 팀의 패수
      "rank": 9,  // 종료 시점의 팀의 랭킹
      "name": "KT"
    },
    "home_team_standing": {
      "draws": 2,
      "era": 5.81,
      "hra": 0.265,
      "wra": 0.369,
      "wins": 24,
      "loses": 41,
      "rank": 10,
      "name": "SAMSUNG"
    }
  },
  ....
]
```