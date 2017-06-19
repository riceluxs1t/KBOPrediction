## Usage
from scrape import *
scraper = DayDataScraper('2017', '05')
interpreter = DayInterpreter('2017', '05')
may_data = DayInterpreter.extract(scraper.parse_raw_data(scraper.get_raw_page()))

