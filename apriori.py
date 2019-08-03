from collections import defaultdict

from utils import generate_candidates, get_all_subsets


class Apriori:
    def __init__(self):
        self.support = {}
        self.rules = defaultdict(list)
        self.transactions = None
        self.items = None
        self.min_support = None

    def fit(self, transactions, items=None, min_support=0.1, min_confidence=0.5, min_lift=1):
        self.load_transactions(transactions, items)
        self.calculate_support(min_support=min_support)
        self.generate_rules(min_confidence=min_confidence, min_lift=min_lift)

    def calculate_support(self, min_support=0.1):
        self.min_support = min_support

        k_candidates = {frozenset([item]) for item in self.items}

        while k_candidates:
            layer_k = []

            for candidate in k_candidates:
                support = sum([candidate.issubset(transaction) for transaction in self.transactions.values()]) / len(
                    self.transactions)

                if support >= min_support:
                    self.support[candidate] = support
                    layer_k.append(candidate)
            k_candidates = generate_candidates(k_candidates=layer_k)

        return len(self.support)

    def generate_rules(self, min_confidence=0.5, min_lift=1):
        for itemset in self.support.keys():
            for subset in get_all_subsets(itemset):
                from_itemset = frozenset(subset)
                to_itemset = itemset.difference(from_itemset)
                confidence = self.support[itemset] / self.support[to_itemset]
                lift = self.support[itemset] / (self.support[to_itemset] * self.support[from_itemset])

                if confidence >= min_confidence and lift >= min_lift:
                    self.rules[from_itemset].append({
                        'to': to_itemset,
                        'confidence': confidence,
                        'lift': lift
                    })
        return len(self.rules)

    def load_transactions(self, transactions, items=None):
        self.transactions = transactions

        if items:
            self.items = items
        else:
            items = set()
            for transaction_items in transactions.values():
                for item in transaction_items:
                    items.add(item)
            self.items = items

    def predict(self, itemset):
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
