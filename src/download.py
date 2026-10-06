from pathlib import Path
from urllib.request import urlretrieve
URL='https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv'
if __name__=='__main__':
    target=Path(__file__).resolve().parents[1]/'data/hotels.csv'
    target.parent.mkdir(exist_ok=True)
    urlretrieve(URL,target)
    print(target)
