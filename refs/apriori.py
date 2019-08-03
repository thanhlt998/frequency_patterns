# -*- coding: utf-8 -*-
"""
    pyfreqpm.apriori
    ~~~~~~~~~~~~~~
    Implements apriori (GEN, TID) algorithm for transactions.
    :copyright: (c) 2016 by Alexey Smirnov.
    :license: BSD, see LICENSE for more details.
"""
from itertools import chain, combinations
from .base import AnalyzerBase, powerset


class AprioriGen(AnalyzerBase):

    def _calc_support(self, min_support=0.1):
        """
        Populates support dict
        Calculate support values for frequent itemsets.
        :param min_support controls what items to check (leaves out rare ones)
        :return: number of supports calculated.
        """
        self.min_support = min_support
        candidate_k = {frozenset([i]) for i in self.items}
        while candidate_k:
            layer_k = []
            for c in candidate_k:
                support = (sum([c.issubset(set(i)) for i in self.transactions]))/len(self.transactions)
                if support >= self.min_support:
                    self.support[c] = support
                    layer_k.append(c)
            candidate_k = {i.union(j) for i in layer_k for j in layer_k if i != j}
        return len(self.support)

    def _gen_rules(self, min_confidence=0.5, min_lift=1.0):
        """
        populate association rules
        :param min_confidence: This says how likely item Y is purchased when item X is purchased,
        expressed as {X -> Y}. This is measured by the proportion of transactions with item X,
        in which item Y also appears.
        :param min_lift: This says how likely item Y is purchased when item X is purchased,
        while controlling for how popular item Y is. Better when >1.
        :return: number of rules generated.
        """
        for itemset in self.support:
            for s in powerset(itemset):
                from_itemset = frozenset(s)
                to_itemset = itemset.difference(s)
                confidence = self.support[itemset]/self.support[from_itemset]
                lift = self.support[itemset]/(self.support[from_itemset]*self.support[to_itemset])
                # prune confidence c{ABC -> D} >= c[AB -> CD} >= c{A -> CD}
                if confidence >= min_confidence and lift >= min_lift:
                    self.rules[from_itemset].append({'to': to_itemset,
                                                     'confidence': confidence,
                                                     'lift': lift})
        return len(self.rules)


class AprioriTid(AprioriGen):
    def _calc_support(self, min_support=0.1):
        """
        Populates support dict
        Calculate support values for frequent itemsets.
        :return: number of supports calculated.
        """
        candidate_dash_k = [set([frozenset([j]) for j in i]) for i in self.transactions]
        self.min_support = min_support
        candidate_k = {frozenset([i]) for i in self.items}
        while candidate_k:
            layer_k = []
            for c in candidate_k:
                #support = (sum([c.issubset(set(i)) for i in self.transactions]))/len(self.transactions)
                support = (sum([c in i for i in candidate_dash_k]))/len(self.transactions) #len(candidate_dash_k)
                if support >= self.min_support:
                    self.support[c] = support
                    layer_k.append(c)
            candidate_k = {i.union(j) for i in layer_k for j in layer_k if i != j}
            temp_candidate_dash_k = [frozenset.union(*i) for i in candidate_dash_k if i]
            candidate_dash_k = []
            for c_dash in temp_candidate_dash_k:
                temp_set = set([c for c in candidate_k if c.issubset(c_dash)])
                candidate_dash_k.append(temp_set)
        return len(self.support)