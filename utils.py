import pandas as pd
from itertools import chain, combinations


def read_data(fn):
    xls = pd.ExcelFile(fn)
    sheet_name = xls.sheet_names[0]
    online_retail = xls.parse(sheet_name=sheet_name)

    transactions = online_retail.loc[:, ['InvoiceNo', 'StockCode']].groupby(by=['InvoiceNo'])['StockCode'].apply(
        lambda x: x.tolist()).to_dict()
    items = set(online_retail.loc[:, 'StockCode'].to_list())

    return transactions, items


def generate_candidates(k_candidates):
    if not k_candidates:
        return {}
    else:
        candidates = set()
        k = len(k_candidates[0]) + 1

        for i, candidate_i in enumerate(k_candidates):
            for candidate_j in k_candidates[i + 1:]:
                candidate = candidate_i.union(candidate_j)
                if len(candidate) == k and check_satisfied_candidates(candidate, k_candidates):
                    candidates.add(candidate)
        return candidates


def check_satisfied_candidates(candidate, k_candidates):
    candidate = set(candidate)
    for item in candidate:
        candidate_copy = candidate.copy()
        candidate_copy.remove(item)
        if not frozenset(candidate_copy) in k_candidates:
            return False
    return True


def get_all_subsets(itemset):
    yield from chain.from_iterable(combinations(itemset, length) for length in range(1, len(itemset)))


if __name__ == '__main__':
    print(read_data('Online Retail.xlsx'))
