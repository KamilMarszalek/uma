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
        self.rng = np.random.default_rng()

    def check_stop_condition(
        self,
        targets: np.ndarray,
        features: list[int],
        remaining_depth: int | None,
    ) -> Node | None:
        if np.unique(targets).size == 1:
            return Node(target=targets[0])
        if len(features) == 0:
            return Node(target=self.most_common_label(targets))
        if remaining_depth is not None and remaining_depth <= 0:
            return Node(target=self.most_common_label(targets))

        return None

    def most_common_label(self, targets: np.ndarray) -> L:
        targets = targets.ravel()
        return Counter(targets).most_common(1)[0][0]

    def tournament_selection(
        self, data: np.ndarray, targets: np.ndarray, features: list[int]
    ) -> int | None:
        chosen = list(
            set(
                self.rng.choice(
                    features,
                    size=self.tournament_size,
                    replace=True,
                )
            )
        )
        best_feature = None
        best_gain = -np.inf

        for feature in chosen:
            gain = self.eval_function(data, targets, feature)
            if gain > best_gain:
                best_feature = feature
                best_gain = gain

        return best_feature

    def split_data(
        self, data: np.ndarray, targets: np.ndarray, feature: int
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        raw = get_splits(data, targets, feature)
        return {
            value: (data[data[:, feature] == value], subset_targets)
            for value, subset_targets in raw.items()
        }

    def build_children(
        self,
        splits: dict[int, tuple[np.ndarray, np.ndarray]],
        features: list[int],
        remain_depth: int | None,
        parent_targets: np.ndarray,
        chosen_feature: int,
    ) -> dict[int, Node]:
        new_features = [f for f in features if f != chosen_feature]
        new_depth = remain_depth - 1 if remain_depth is not None else None

        children: dict[int, Node] = {}

        for value, (subset_data, subset_targets) in splits.items():
            if subset_targets.size == 0:
                children[value] = Node(
                    target=self.most_common_label(parent_targets),
                )
                continue

            children[value] = self.build_tree(
                subset_data,
                subset_targets,
                new_features,
                remaining_depth=new_depth,
            )

        return children

    def build_tree(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        features: list[int],
        remaining_depth: int | None = None,
    ) -> Node:
        stop_node = self.check_stop_condition(
            targets,
            features,
            remaining_depth,
        )
        if stop_node is not None:
            return stop_node

        chosen_feature = self.tournament_selection(data, targets, features)
        if chosen_feature is None:
            return Node(target=self.most_common_label(targets))

        splits = self.split_data(data, targets, chosen_feature)
        children = self.build_children(
            splits,
            features,
            remaining_depth,
            targets,
            chosen_feature,
        )

        return Node(
            feature=chosen_feature,
            children=children,
            default_label=self.most_common_label(targets),
        )

    def predict(self, tree: Node, sample: np.ndarray) -> L:
        if tree.target is not None:
            return tree.target
        value = sample[tree.feature]
        child = tree.children.get(value)
        if child is None:
            return tree.default_label
        return self.predict(child, sample)
