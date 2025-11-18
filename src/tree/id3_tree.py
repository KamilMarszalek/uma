from typing import Any

import numpy as np

from src.tree.base_tree import BaseTree
from src.tree.config import ID3Config
from src.tree.get_splits import get_splits
from src.tree.node import Node


class ID3Tree(BaseTree):
    def __init__(
        self,
        config: ID3Config,
    ) -> None:
        super().__init__()
        self.eval_function = config.eval_function
        self.tournament_size = config.tournament_size

    def fit(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        features: list[int],
    ) -> None:
        self.root = self.build_tree(
            data,
            targets,
            features,
            remaining_depth=None,
        )

    def predict(self, sample: np.ndarray) -> Any:
        if self.root is None:
            raise ValueError("The tree has not been fitted yet.")
        return self._predict(self.root, sample)

    def check_stop_condition(
        self,
        targets: np.ndarray,
        features: list[int],
        remaining_depth: int | None,
    ) -> Node | None:
        if np.unique(targets).size == 1:
            return Node(target=targets[0])
        if len(features) == 0:
            return Node(target=BaseTree.most_common_label(targets))
        if remaining_depth is not None and remaining_depth <= 0:
            return Node(target=BaseTree.most_common_label(targets))

        return None

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
        col = data[:, feature]

        splits: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for value, subset_targets in raw.items():
            mask = col == value
            subset_data = data[mask]
            splits[value] = (subset_data, subset_targets)

        return splits

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
                    target=BaseTree.most_common_label(parent_targets),
                )
                continue

            if len(new_features) == 0:
                children[value] = Node(
                    target=BaseTree.most_common_label(subset_targets),
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
            return Node(target=BaseTree.most_common_label(targets))

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
            default_label=BaseTree.most_common_label(targets),
        )

    def _predict(self, tree: Node, sample: np.ndarray) -> Any:
        if tree.target is not None:
            return (
                tree.target.item()
                if isinstance(tree.target, np.ndarray)
                else tree.target
            )

        if tree.children is None:
            return tree.default_label

        value = int(sample[tree.feature])
        child = tree.children.get(value)
        if child is None:
            return tree.default_label

        return self._predict(child, sample)
