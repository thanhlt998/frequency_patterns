from .base import AnalyzerBase, powerset
from .apriori import AprioriGen
from collections import Counter, defaultdict, deque
from itertools import dropwhile
from pprint import pprint
from itertools import combinations

class FPGrowth(AprioriGen):
    """ FP-growth algorithm
    array for tree representation
    """
    def _calc_support(self, min_support=0.1):
        self.min_support = min_support
        fp_tree, head = self._generate_tree2()
        pq = deque()
        pq.append((fp_tree, head, [None]))
        while pq:
            q = pq.popleft()

            fp_tree_proj, head_proj = self._generate_proj_tree(q[0], q[1], q[2][-1])
            if len(fp_tree_proj) > 1:
                if all([len(i['children']) <= 1 for i in fp_tree_proj]):
                    for i in range(1, len(fp_tree_proj)):
                        for s in combinations(fp_tree_proj[1:], i):
                            itemset = frozenset([x['item'] for x in s if x['item']] + q[2][1:])
                            min_support = min([x['support'] for x in s if x['item']])
                            self.support[itemset] = min_support

                else:
                    for i in head_proj:
                        itemset = q[2].copy()
                        itemset.append(i)
                        pq.append((fp_tree_proj, head_proj, itemset))
                        self.support[frozenset(itemset[1:])] = sum([fp_tree_proj[j]['support'] for j in head_proj[i]])
        l = len(self.transactions)
        for i in self.support:
            self.support[i] /= l
        return len(self.support)

    def _generate_proj_tree(self, fp_tree, head, item):
        if not item:
            return fp_tree, head
        fp_tree_proj = []
        head_proj = defaultdict(list)
        root = {'parent': None, 'children': [], 'item': None, 'support': 0}
        fp_tree_proj.append(root)
        for i in head[item]:
            pointer = None
            support = fp_tree[i]['support']
            parent = fp_tree[fp_tree[i]['parent']].copy()
            if not parent['item']:
                parent = None

            # go up by conditional path
            while parent:
                parent['children'] = []
                parent['support'] = support

                # node with this item was seen somewhere on the tree. try to merge
                head_pointer = -1
                for i in list(head_proj[parent['item']]):
                    if fp_tree_proj[i]['parent'] == parent['parent']:
                        head_pointer = i

                # merge path by node
                if head_pointer > 0:
                    fp_tree_proj[i]['support'] += parent['support']
                    if pointer:
                        fp_tree_proj[head_pointer]['children'].append(pointer)
                    pointer = None

                # new node
                else:
                    fp_tree_proj.append(parent)
                    head_proj[parent['item']].append(len(fp_tree_proj) - 1)
                    if pointer:
                        parent['children'].append(pointer)
                    pointer = len(fp_tree_proj) - 1
                    if parent['parent'] == 0:
                        fp_tree_proj[0]['children'].append(pointer)

                # proceed further
                if parent['parent']:
                    parent = fp_tree[parent['parent']].copy()
                else:
                    parent = None

        # fix parent
        for n, i in enumerate(fp_tree_proj):
            if i['item']:
                for c in i['children']:
                    fp_tree_proj[c]['parent'] = n

        # prune by support
        for i in head_proj:
            if sum([fp_tree_proj[x]['support'] for x in head_proj[i]]) < self.min_support_count:
                for p in head_proj[i]:
                    fp_tree_proj[fp_tree_proj[p]['parent']]['children'].remove(p)
                    for c in fp_tree_proj[p]['children']:
                        fp_tree_proj[c]['parent'] = fp_tree_proj[p]['parent']
                        fp_tree_proj[fp_tree_proj[p]['parent']]['children'].append(c)
                    fp_tree_proj[p]['support'] = -1
                head_proj[i] = []

        # ensure tree consistency
        cmap = dict()
        fp_tree_proj_temp = []
        for n, i in enumerate(fp_tree_proj):
            if i['support'] >= 0:
                fp_tree_proj_temp.append(i)
            cmap[n] = len(fp_tree_proj_temp) - 1

        # pprint(cmap)
        for i in range(len(fp_tree_proj_temp)):
            if fp_tree_proj_temp[i]['parent']:
                fp_tree_proj_temp[i]['parent'] = cmap[ fp_tree_proj_temp[i]['parent']]
            for c in range(len(fp_tree_proj_temp[i]['children'])):
                fp_tree_proj_temp[i]['children'][c] = cmap[ fp_tree_proj_temp[i]['children'][c]]

        # merge sibling nodes with same items
        q = deque([0])
        while q:
            qq = q.popleft()
            for a, b in combinations(fp_tree_proj_temp[qq]['children'], 2):
                if fp_tree_proj_temp[a]['item'] == fp_tree_proj_temp[b]['item'] and \
                                fp_tree_proj_temp[a]['parent'] == fp_tree_proj_temp[b]['parent']:
                    fp_tree_proj_temp[a]['support'] += fp_tree_proj_temp[b]['support']
                    fp_tree_proj_temp[a]['children'] += fp_tree_proj_temp[b]['children']
                    fp_tree_proj_temp[b]['support'] = -1
                    if b in fp_tree_proj_temp[fp_tree_proj_temp[a]['parent']]['children']:
                        fp_tree_proj_temp[fp_tree_proj_temp[a]['parent']]['children'].remove(b)
            for c in fp_tree_proj_temp[qq]['children']:
                q.append(c)

        # ensure tree consistency #2
        cmap = dict()
        fp_tree_proj_temp2 = []
        for n, i in enumerate(fp_tree_proj_temp):
            if i['support'] >= 0:
                fp_tree_proj_temp2.append(i)
            cmap[n] = len(fp_tree_proj_temp2) - 1

        for i in range(len(fp_tree_proj_temp2)):
            if fp_tree_proj_temp2[i]['parent']:
                fp_tree_proj_temp2[i]['parent'] = cmap[fp_tree_proj_temp2[i]['parent']]
            for c in range(len(fp_tree_proj_temp2[i]['children'])):
                fp_tree_proj_temp2[i]['children'][c] = cmap[fp_tree_proj_temp2[i]['children'][c]]

        # populate head table after all consistency checks
        head_proj2 = defaultdict(list)
        for i in range(len(fp_tree_proj_temp2)):
            if fp_tree_proj_temp2[i]['item']:
                head_proj2[fp_tree_proj_temp2[i]['item']].append(i)

        return fp_tree_proj_temp2, head_proj2


    def _generate_tree2(self):
        """ Generate initial full FP-Tree
        :param min_support: controls what items to check (leaves out rare ones)
        :return: list with fp-tree and head table
        """
        # filter 1-itemsets by min_support
        self.min_support_count = self.min_support * len(self.transactions)
        item_support = Counter([item for sublist in self.transactions for item in sublist])
        for key, count in dropwhile(lambda key_count: key_count[1] >= self.min_support_count, item_support.most_common()):
            del item_support[key]

        # generate initial fp_tree
        fp_tree = []
        root = {'parent': None, 'children': [], 'item': None, 'support': 0}
        fp_tree.append(root)
        head = defaultdict(list)

        for tran in self.transactions:
            tree_pointer = 0
            trans_sorted = sorted([i for i in tran if i in item_support],
                                  key=lambda x: (item_support[x], x), reverse=True)
            for t in trans_sorted:
                current_children = dict([(fp_tree[i]['item'], i) for i in fp_tree[tree_pointer]['children']])
                if t in current_children:
                    index = current_children[t]
                    fp_tree[index]['support'] += 1
                    tree_pointer = index
                else:
                    node = {'parent': tree_pointer, 'children': [], 'item': t, 'support': 1}
                    fp_tree.append(node)
                    fp_tree[tree_pointer]['children'].append(len(fp_tree)-1)
                    tree_pointer = len(fp_tree)-1
                    head[t].append(tree_pointer)
        return fp_tree, head
