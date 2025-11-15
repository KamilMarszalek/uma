from collections import Counter

import numpy as np
from eval_func import EvalFunc
from get_splits import get_splits
from node import L, Node


class Tree:
    def __init__(
        self,
        eval_function: EvalFunc,
        tournament_size: int = 2,
    ) -> None:
        self.eval_function = eval_function
        self.tournament_size = tournament_size

    def most_common_label(self, targets: np.ndarray) -> L | None:
        targets = targets.ravel()
        if targets.size == 0:
            return None
        return Counter(targets).most_common(1)[0][0]

    def tournament_selection(
        self, data: np.ndarray, targets: np.ndarray, features: list[int]
    ) -> int | None:
        rng = np.random.default_rng()
        chosen = list(
            set(rng.choices(features, size=self.tournament_size, replace=True))
        )
        best_feature = None
        best_gain = -np.inf
        for feature in chosen:
            gain = self.eval_function(data, targets, feature)
            if gain > best_gain:
                best_feature = feature
                best_gain = gain
        return best_feature

    def build_tree(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        features: list[int],
        max_depth: int | None = None,
    ) -> "Node":
        if np.unique(targets).size == 1:
            return Node(target=targets[0])
        if len(features) == 0:
            return Node(target=self.most_common_label(targets))
        if max_depth is not None and max_depth <= 0:
            return Node(target=self.most_common_label(targets))
        chosen_feature = self.tournament_selection(data, targets, features)
        if chosen_feature is None:
            return Node(target=self.most_common_label(targets))
        splits = get_splits(data, targets, chosen_feature)
        children = {}
        for value, subset_targets in splits.items():
            subset_data = data[data[:, chosen_feature] == value]
            if subset_targets.size == 0:
                children[value] = Node(target=self.most_common_label(targets))
                continue
            new_features = [feat for feat in features if feat != chosen_feature]
            child_node = self.build_tree(
                subset_data,
                subset_targets,
                new_features,
                max_depth=max_depth - 1
                if max_depth is not None and max_depth <= 0
                else None,
            )
            children[value] = child_node
        return Node(
            feature=chosen_feature,
            children=children,
            default_label=self.most_common_label(targets),
        )

    def predict(self, tree: "Node", sample: np.ndarray) -> L:
        if tree.target is not None:
            return tree.target

        if sample[tree.feature] in tree.children:
            return self.predict(tree.children[sample[tree.feature]], sample)
        else:
            return tree.default_label
