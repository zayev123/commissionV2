from bs4 import BeautifulSoup
import requests
from lxml import html

class StockParser():
    def __init__(self):
        self.url = 'https://dps.psx.com.pk/trading-board/REG/main'

    def parse(self):
        response = requests.get(self.url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, 'html.parser')
            # cls1 = soup.find_all(attrs={"id": "tradingPanelBoard"})
            table = soup.find('table', attrs={"id": "tradingBoardTable"})
            rows = table.find_all('tr')
            data_rows = []
            for row in rows:
                # Extract and print the contents of each cell in the row
                cells = row.find_all('td')
                row_data = [cell.text.strip() for cell in cells]
                data_rows.append(row_data)
            return data_rows
        else:
            print('Failed to retrieve the webpage. Status code:', response.status_code)
            return None