## Usage
from scrape import *

scraper = DayDataScraper('2017', '05')

interpreter = DayInterpreter('2017', '05')

may_data = interpreter.extract(scraper.parse_raw_page(scraper.get_raw_page()))

