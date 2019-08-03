from apriori import Apriori
from fpgrowth import FPGrowth
from utils import read_data
import time

if __name__ == '__main__':
    transactions, items = read_data('Online Retail.xlsx')
    # transactions, items = {'T1': ['pasta', 'lemon', 'bread', 'orange'],
    #                        'T2': ['pasta', 'lemon'],
    #                        'T3': ['pasta', 'orange', 'cake'],
    #                        'T4': ['pasta', 'lemon', 'orange', 'cake']}, ['pasta', 'lemon', 'bread', 'orange', 'cake']

    t1 = time.time()
    apriori = Apriori()
    apriori.fit(transactions=transactions, items=items, min_support=0.03)
    # result = apriori.predict(['pasta', 'lemon'])
    result = apriori.predict(['84029G', '84029E'])
    print(len(apriori.rules))
    t2 = time.time()
    print(result)
    print(t2 - t1)

    print('--------------------------------------------')

    t1 = time.time()
    fp_growth = FPGrowth()
    fp_growth.fit(transactions=transactions, items=items, min_support=0.03)
    result = fp_growth.predict(['84029G', '84029E'])
    # result = fp_growth.predict(['pasta', 'lemon'])
    print(len(fp_growth.rules))
    t2 = time.time()
    print(result)
    print(t2 - t1)

