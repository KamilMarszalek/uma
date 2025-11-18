import numpy as np

from src.forest.config import TournamentForestConfig
from src.forest.forest import TournamentForest
from src.tree.eval_func import InformationGain
from src.tree.config import ID3Config
from src.tree.cart_tree import ID3Tree


def test_forest_predict():
    data = np.array([[0], [0], [1], [1]])
    targets = np.array([0, 0, 1, 1])

    config = TournamentForestConfig(
        num_of_trees=5,
        sample_ratio=1.0,
        feature_ratio=1.0,
        tree_class=ID3Tree,
        tree_config_class=ID3Config,
        eval_function=InformationGain(),
        max_depth=3,
        tournament_size=2,
    )
    forest = TournamentForest(data, targets, config)
    forest.fit()

    assert forest.predict(np.array([0])) == 0
    assert forest.predict(np.array([1])) == 1


def test_forest_bootstrap_sampling():
    data = np.arange(100).reshape(100, 1)
    targets = np.zeros(100)

    config = TournamentForestConfig(
        num_of_trees=3,
        sample_ratio=0.5,
        feature_ratio=1.0,
        eval_function=InformationGain(),
        tree_class=ID3Tree,
        tree_config_class=ID3Config,
        max_depth=3,
        tournament_size=2,
    )
    forest = TournamentForest(data, targets, config)
    forest.fit()

    # forest should contain 3 trees
    assert len(forest.forest) == 3

    # Each tree should get subset of rows
    for tree in forest.forest:
        assert tree.root is not None
