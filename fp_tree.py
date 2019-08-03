from collections import defaultdict, Counter

from utils import get_all_subsets


class Node:
    def __init__(self, item_name, item_count=0, parent=None, children=None):
        self.item_name = item_name
        self.item_count = item_count
        self.parent = parent
        self.children = children if children else {}
        self.path_to_root = self.get_path_to_root()

    def get_path_to_root(self):
        if self.parent is None or self.parent.item_name is None:
            return []
        else:
            return [*self.parent.path_to_root, self.parent.item_name]

    def __eq__(self, other):
        return self.item_name == other.item_name


class FPTree:
    def __init__(self, transactions, min_support_count):
        self.min_support_count = min_support_count
        self.transactions = transactions
        self.head = defaultdict(list)
        self.root = Node(None)
        self.support = {}

        self.item_support_count = defaultdict(lambda: 0)
        for items, count in transactions:
            for item in items:
                self.item_support_count[item] += count
        self.item_support_count = {item: count for item, count in self.item_support_count.items() if
                                   count >= self.min_support_count}
        self.frequency_list = list(self.item_support_count.keys())

        self.generate_tree()

    def generate_tree(self):
        for items, count in self.transactions:
            sorted_items = sorted(
                [(item, self.item_support_count[item]) for item in items if item in self.item_support_count],
                key=lambda x: x[1],
                reverse=True)
            current = self.root
            for item, _ in sorted_items:
                if item in current.children:
                    current.children[item].item_count += count
                    current = current.children[item]
                else:
                    item_node = Node(item, item_count=count, parent=current)
                    self.head[item].append(item_node)
                    current.children[item] = item_node
                    current = item_node

    def get_conditional_pattern_base(self):
        conditional_pattern_bases = defaultdict(list)
        for item in self.frequency_list[::-1]:
            for project_node in self.head[item]:
                conditional_pattern_bases[item].append((project_node.path_to_root, project_node.item_count))

        return conditional_pattern_bases

    def get_conditional_frequent_pattern(self):

        if len(self.root.children) == 1 and len(list(self.root.children.values())[0].children) == 0:
            return [([list(self.root.children.values())[0].item_name], list(self.root.children.values())[0].item_count)]

        conditional_pattern_bases = self.get_conditional_pattern_base()
        new_conditional_pattern_bases = []
        for item, conditional_pattern_base in conditional_pattern_bases.items():
            new_conditional_pattern_bases.append(([item], self.item_support_count[item]))
            fp_tree = FPTree(conditional_pattern_base, self.min_support_count)
            for items, count in fp_tree.get_conditional_frequent_pattern():
                new_conditional_pattern_bases.append(([*items, item], count))

        return new_conditional_pattern_bases

    # def generate_fp(self, conditional_fp_tree):
    #     for item, conditional_item_count in conditional_fp_tree.items():
    #         for subset in get_all_subsets(conditional_item_count.keys()):
    #             self.support[frozenset([*subset, item])] = min(
    #                 [conditional_item_count[subset_item]] for subset_item in subset)
