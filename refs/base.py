from itertools import chain, combinations
from collections import defaultdict
from abc import ABCMeta, abstractmethod


def powerset(iterable):
    """
    Returns all subsets of a set
    :param iterable: set
    :return: generator over all subsets
    """
    s = list(iterable)
    yield from chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


class AnalyzerGen:
    def __init__(self):
        raise NotImplementedError


class AnalyzerBase(metaclass=ABCMeta):
    def __init__(self):
        self.support = {}
        self.rules = defaultdict(list)

    def _load_transactions(self, transactions, items):
        """
        Analyze transactions and builds overall itemset
        :param transactions: iterable over transactions
        :param items: a iterable over unique items that can be found in transactions
        :return: number of items found
        """
        self.transactions = list(transactions)
        if not items:
            self.items = list({i for i in chain.from_iterable(transactions)})
        else:
            self.items = items
        return len(self.items), len(self.transactions)

    @abstractmethod
    def _calc_support(self, min_support=0.1):
        """Populates support dict"""

    @abstractmethod
    def _gen_rules(self, min_confidence=0.5, min_lift=1.0):
        """populate association rules"""

    def fit(self, transactions, items=None, min_support=0.1, min_confidence=0.5, min_lift=1.0):
        """
        External interface to load transaction into memmory and generate rules.
        :param transactions: iterable over transactions
        :param items: a iterable over unique items that can be found in transactions
        :param min_support: controls what items to check (leaves out rare ones)
        :param min_confidence: This says how likely item Y is purchased when item X is purchased,
        expressed as {X -> Y}. This is measured by the proportion of transactions with item X,
        in which item Y also appears.
        :param min_lift: This says how likely item Y is purchased when item X is purchased,
        while controlling for how popular item Y is. Better when >1.
        :return: dict with counts and generated rules
        """
        items_count, transactions_count = self._load_transactions(transactions, items)
        return {'items_count': items_count,
                'transactions_count': transactions_count,
                'support_count': self._calc_support(min_support),
                'rule_count': self._gen_rules(min_confidence=min_confidence, min_lift=min_lift)}

    def predict(self, itemset):
        """
        External interface to find recommendadtion for a basket containing itemset
        :param itemset: items in basket
        :return: best rule
        """
        itemset = frozenset(itemset)
        max_confidence = 0
        max_lift = 0
        max_rule = None
        if itemset in self.rules:
            for rule in self.rules[itemset]:
                if rule['confidence'] >= max_confidence and rule['lift'] >= max_lift:
                    max_confidence = rule['confidence']
                    max_lift = rule['lift']
                    max_rule = rule
        return max_rule