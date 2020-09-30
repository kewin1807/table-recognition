# sample crawl data
import bs4
import pandas as pd
import lxml.html as lh

import requests
url = "http://www.espn.com/sports/tennis/rankings"


def get_page_content(url):
    page = requests.get(url)

    doc = lh.fromstring(page.content)
    print(doc)
    tr_elements = doc.xpath('//tr')

    # Create empty list
    col = []
    i = 0
    for t in tr_elements[1]:
        i += 1
        name = t.text_content()
        print('%d:"%s"' % (i, name))
        col.append((name, []))
    for j in range(2, len(tr_elements)):
        # T is our j'th row
        T = tr_elements[j]

    # If row is not of size 10, the //tr data is not from our table
        if len(T) != 5:
            break

    # i is the index of our column
        i = 0

    # Iterate through each element of the row
        for t in T.iterchildren():
            data = t.text_content()
        # Check if row is empty
            if i > 0:
                # Convert any numerical value to integers
                try:
                    data = int(data)
                except:
                    pass
        # Append the data to the empty list of the i'th column
            col[i][1].append(data)
        # Increment i for the next column
            i += 1

    Dict = {title: column for (title, column) in col}
    df = pd.DataFrame(Dict)
    print(df)


get_page_content(url)
