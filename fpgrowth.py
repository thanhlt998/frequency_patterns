from fp_tree import FPTree
from apriori import Apriori


class FPGrowth(Apriori):
    def __init__(self):
        super(FPGrowth, self).__init__()

    # def fit(self, transactions, items=None, min_support=0.1, min_confidence=0.5, min_lift=1):

    def calculate_support(self, min_support=0.1):
        fp_tree = FPTree(transactions=tuple(zip(self.transactions.values(), [1] * len(self.transactions))),
                         min_support_count=min_support * len(self.transactions))
        no_transactions = len(self.transactions)
        fp = fp_tree.get_conditional_frequent_pattern()
        for itemset, count in fp:
            self.support[frozenset(itemset)] = count / no_transactions

        return len(self.support)
