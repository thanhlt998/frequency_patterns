from .apriori import AprioriGen
from collections import defaultdict
from pprint import pprint


class Eclat(AprioriGen):
    def _calc_support(self, min_support=0.1):
        """
        Populates support dict
        Calculate support values for frequent itemsets.
        :return: number of supports calculated.
        """
        self.min_support = min_support
        candidate_k = defaultdict(set)
        transaction_length = len(self.transactions)

        for n, transaction in enumerate(self.transactions):
            for item in transaction:
                candidate_k[frozenset([item])].add(n)

        for item in self.items:
            support = len(candidate_k[frozenset([item])]) / transaction_length
            if support >= self.min_support:
                self.support[frozenset([item])] = len(candidate_k[frozenset([item])])/transaction_length

        # pprint(self.support)
        while candidate_k:
            candidate_k_temp = defaultdict(set)
            for c in candidate_k:
                for cc in candidate_k:
                    union = c | cc
                    if c != cc and union not in candidate_k_temp:
                        candidate_k_temp[union] = candidate_k[c] & candidate_k[cc]
            for c in candidate_k_temp:
                support = len(candidate_k_temp[c])/transaction_length
                if support >= self.min_support:
                    self.support[c] = support

            candidate_k = candidate_k_temp

        return len(self.support)