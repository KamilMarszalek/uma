from collections import Counter
from enum import Enum
from typing import Any

import numpy as np

from src.tree.config import CARTConfig, ID3Config
from src.tree.get_splits import get_splits
from src.tree.node import Node


class ID3Tree:
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        features: list[int],
        config: ID3Config,
    ) -> None:
        self.eval_function = config.eval_function
        self.tournament_size = config.tournament_size
        self.rng = np.random.default_rng()
        self.root = self.build_tree(
            data,
            targets,
            features,
            config.max_depth,
        )

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

    def most_common_label(self, targets: np.ndarray) -> Any:
        targets = targets.ravel()
        if targets.size == 0:
            return None
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
                    target=self.most_common_label(parent_targets),
                )
                continue

            if len(new_features) == 0:
                children[value] = Node(
                    target=self.most_common_label(subset_targets),
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

    def predict(self, sample: np.ndarray) -> Any:
        return self._predict(self.root, sample)

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


class CARTTree:
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        features: list[int],
        config: CARTConfig,
    ) -> None:
        self.max_depth = config.max_depth
        self.min_samples_split = config.min_samples_split
        self.tournament_size = config.tournament_size
        self.eval_function = config.eval_function
        self.rng = np.random.default_rng()

        self.root = self.build_tree(
            data,
            targets,
            features,
            remaining_depth=self.max_depth,
        )

    def check_stop_condition(
        self,
        targets: np.ndarray,
        remaining_depth: int | None,
    ) -> Node | None:
        if targets.size < self.min_samples_split:
            return Node(target=self.most_common_label(targets))
        if np.unique(targets).size == 1:
            return Node(target=targets[0])
        if remaining_depth is not None and remaining_depth <= 0:
            return Node(target=self.most_common_label(targets))
        return None

    def most_common_label(self, targets: np.ndarray) -> Any:
        if targets.size == 0:
            return None
        return Counter(targets).most_common(1)[0][0]

    def tournament_selection(
        self, data: np.ndarray, targets: np.ndarray, features: list[int]
    ) -> tuple[int | None, float | None]:
        candidates = list(
            set(
                self.rng.choice(
                    features,
                    size=self.tournament_size,
                    replace=True,
                )
            )
        )

        best_feature = None
        best_thr = None
        best_gain = -np.inf

        for feature in candidates:
            thr, gain = self.find_best_threshold(data[:, feature], targets)
            if gain > best_gain:
                best_gain = gain
                best_thr = thr
                best_feature = feature

        return best_feature, best_thr

    def find_best_threshold(
        self, col: np.ndarray, targets: np.ndarray
    ) -> tuple[float | None, float]:
        values = np.unique(col)
        if values.size <= 1:
            return None, -np.inf

        values.sort()
        thresholds = (values[:-1] + values[1:]) / 2

        best_gain = -np.inf
        best_thr = None

        for thr in thresholds:
            left = targets[col <= thr]
            right = targets[col > thr]

            gain = self.eval_function(targets, left, right)

            if gain > best_gain:
                best_gain = gain
                best_thr = thr

        return best_thr, best_gain

    def build_tree(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        features: list[int],
        remaining_depth: int | None,
    ) -> Node:
        stop = self.check_stop_condition(
            targets,
            remaining_depth,
        )
        if stop is not None:
            return stop
        chosen = self.tournament_selection(data, targets, features)
        if chosen is None:
            return Node(target=self.most_common_label(targets))
        feature, thr = chosen
        if feature is None or thr is None:
            return Node(target=self.most_common_label(targets))
        col = data[:, feature]
        left_mask = col <= thr
        right_mask = col > thr
        children = {
            "le": self.build_tree(
                data[left_mask],
                targets[left_mask],
                features,
                remaining_depth - 1 if remaining_depth is not None else None,
            ),
            "gt": self.build_tree(
                data[right_mask],
                targets[right_mask],
                features,
                remaining_depth - 1 if remaining_depth is not None else None,
            ),
        }
        return Node(
            feature=feature,
            threshold=thr,
            children=children,
            default_label=self.most_common_label(targets),
        )

    def predict(self, sample: np.ndarray) -> Any:
        node = self.root
        while node.target is None:
            value = sample[node.feature]
            key = "le" if value <= node.threshold else "gt"
            if not node.children:
                return node.default_label
            if key not in node.children:
                return node.default_label
            node = node.children[key]
        return node.target


class TreeClass(Enum):
    ID3 = ID3Tree
    CART = CARTTree

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
