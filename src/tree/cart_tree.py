from typing import Any

import numpy as np

from src.tree.base_tree import BaseTree
from src.tree.config import CARTConfig
from src.tree.node import Node


class CARTTree(BaseTree):
    def __init__(
        self,
        config: CARTConfig,
    ) -> None:
        super().__init__(config.random_seed)
        self.max_depth = config.max_depth
        self.min_samples_split = config.min_samples_split
        self.tournament_size = config.tournament_size
        self.eval_function = config.eval_function

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
            remaining_depth=self.max_depth,
        )

    def predict(self, sample: np.ndarray) -> Any:
        node = self.root
        if node is None:
            raise ValueError("The tree has not been fitted yet.")
        while node.target is None:
            value = sample[node.feature]
            key = "le" if value <= node.threshold else "gt"
            if not node.children:
                return node.default_label
            if key not in node.children:
                return node.default_label
            node = node.children[key]
        return node.target

    def check_stop_condition(
        self,
        targets: np.ndarray,
        remaining_depth: int | None,
    ) -> Node | None:
        if targets.size < self.min_samples_split:
            return Node(target=BaseTree.most_common_label(targets))
        if np.unique(targets).size == 1:
            return Node(target=targets[0])
        if remaining_depth is not None and remaining_depth <= 0:
            return Node(target=BaseTree.most_common_label(targets))
        return None

    def tournament_selection(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        features: list[int],
    ) -> tuple[int | None, float | None]:
        parent_total = targets.size
        parent_pos = int(targets.sum())
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
            thr, gain = self.find_best_threshold(
                data[:, feature],
                targets,
                parent_pos=parent_pos,
                parent_total=parent_total,
            )
            if gain > best_gain:
                best_gain = gain
                best_thr = thr
                best_feature = feature
        return best_feature, best_thr

    def find_best_threshold(
        self,
        col: np.ndarray,
        targets: np.ndarray,
        *,
        parent_pos: int,
        parent_total: int,
    ) -> tuple[float | None, float]:
        if parent_total <= 1:
            return None, -np.inf
        sorted_indices = np.argsort(col, kind="mergesort")
        sorted_col = col[sorted_indices]
        sorted_targets = targets[sorted_indices]
        different_values = sorted_col[1:] != sorted_col[:-1]
        split_pos = np.nonzero(different_values)[0]
        if split_pos.size == 0:
            return None, -np.inf
        pos_cum = np.cumsum(sorted_targets, dtype=np.int32)
        left_total = (split_pos + 1).astype(np.int32)
        left_pos = pos_cum[split_pos].astype(np.int32)
        gains = self.eval_function(
            parent_pos=parent_pos,
            parent_total=parent_total,
            left_pos=left_pos,
            left_total=left_total,
        )
        best_index = int(np.argmax(gains))
        indices = int(split_pos[best_index])
        threshold = float(
            (sorted_col[indices] + sorted_col[indices + 1]) / 2.0,
        )
        best_gain = float(gains[best_index])
        return threshold, best_gain

    def build_tree(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        features: list[int],
        remaining_depth: int | None = None,
    ) -> Node:
        stop = self.check_stop_condition(
            targets,
            remaining_depth,
        )
        if stop is not None:
            return stop
        chosen = self.tournament_selection(data, targets, features)
        if chosen is None:
            return Node(target=BaseTree.most_common_label(targets))
        feature, thr = chosen
        if feature is None or thr is None:
            return Node(target=BaseTree.most_common_label(targets))
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
            default_label=BaseTree.most_common_label(targets),
        )
