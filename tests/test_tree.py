import numpy as np

from src.tree.config import ID3Config
from src.tree.eval_func import InformationGain
from src.tree.id3_tree import ID3Tree


def test_tree_single_label():
    data = np.array([[1], [1], [1]])
    targets = np.array([0, 0, 0])
    config = ID3Config(
        max_depth=3,
        eval_function=InformationGain(),
        tournament_size=2,
    )

    tree = ID3Tree(config=config)
    tree.fit(
        data=data,
        targets=targets,
        features=[0],
    )
    assert tree.root.target == 0


def test_tree_max_depth_stops():
    data = np.array([[1], [2], [3]])
    targets = np.array([0, 1, 1])
    config = ID3Config(
        max_depth=0,
        eval_function=InformationGain(),
        tournament_size=2,
    )

    tree = ID3Tree(config=config)
    tree.fit(
        data=data,
        targets=targets,
        features=[0],
    )
    assert tree.root.target in {0, 1}  # majority or some valid prediction


def test_tree_stops_when_no_features():
    data = np.array([[1], [2], [3]])
    targets = np.array(["A", "B", "A"])
    tree_config = ID3Config(
        max_depth=5, eval_function=InformationGain(), tournament_size=2
    )
    tree = ID3Tree(config=tree_config)
    tree.fit(
        data=data,
        targets=targets,
        features=[],
    )

    assert tree.root.target == "A"  # majority
    assert tree.root.children is None


def test_tree_splits_correctly():
    # Feature perfectly separates classes
    data = np.array([[0], [0], [1], [1]])
    targets = np.array([0, 0, 1, 1])

    config = ID3Config(
        max_depth=3,
        eval_function=InformationGain(),
        tournament_size=2,
    )
    tree = ID3Tree(config=config)
    tree.fit(
        data=data,
        targets=targets,
        features=[0],
    )

    assert tree.root.feature == 0
    assert 0 in tree.root.children
    assert 1 in tree.root.children
    assert tree.root.children[0].target == 0
    assert tree.root.children[1].target == 1


def test_tree_predict():
    data = np.array([[0], [0], [1], [1]])
    targets = np.array([0, 0, 1, 1])

    config = ID3Config(
        max_depth=3,
        eval_function=InformationGain(),
        tournament_size=2,
    )
    tree = ID3Tree(config=config)
    tree.fit(
        data=data,
        targets=targets,
        features=[0],
    )

    assert tree.predict(np.array([0])) == 0
    assert tree.predict(np.array([1])) == 1


def make_tree(data, targets, features=None, max_depth=5):
    if features is None:
        features = list(range(data.shape[1]))
    config = ID3Config(
        max_depth=max_depth,
        eval_function=InformationGain(),
        tournament_size=2,
    )
    tree = ID3Tree(config=config)
    tree.fit(
        data=data,
        targets=targets,
        features=features,
    )
    return tree


def test_tree_handles_mixed_labels():
    data = np.array([[0], [0], [1], [1]])
    targets = np.array(["yes", "yes", "no", "no"])
    tree = make_tree(data, targets, features=[0])

    assert tree.root.feature == 0
    assert tree.predict(np.array([0])) == "yes"
    assert tree.predict(np.array([1])) == "no"


def test_tree_predict_uses_default_label_when_missing_child():
    data = np.array([[0], [1]])
    targets = np.array([0, 1])
    tree = make_tree(data, targets)

    assert tree.predict(np.array([2])) in {0, 1}


def test_tree_with_two_features_splits_on_best():
    data = np.array(
        [
            [0, 10],
            [0, 11],
            [1, 20],
            [1, 21],
        ]
    )
    targets = np.array([0, 0, 1, 1])

    tree = make_tree(data, targets, features=[0, 1])

    assert tree.root.feature in {0, 1}

    assert set(tree.root.children.keys()) == {0, 1} or set(
        tree.root.children.keys()
    ) == {10, 11, 20, 21}


def test_tree_depth_limit_works_properly():
    data = np.array([[0], [0], [1], [1], [2], [2]])
    targets = np.array([0, 0, 1, 1, 0, 0])
    tree = make_tree(data, targets, max_depth=0)

    assert tree.root.target in {0, 1}


def test_tree_tournament_selection_restricted_features():
    data = np.array([[0], [1], [0], [1]])
    targets = np.array([0, 1, 0, 1])

    config = ID3Config(
        max_depth=3,
        eval_function=InformationGain(),
        tournament_size=10,
    )
    tree = ID3Tree(config=config)
    tree.fit(data=data, targets=targets, features=[0])
    assert tree.root.feature == 0


def test_tree_handles_feature_with_single_unique_value():
    data = np.array([[5], [5], [5], [5]])
    targets = np.array([1, 1, 1, 1])
    tree = make_tree(data, targets)

    assert tree.root.target == 1
    assert tree.root.children is None or len(tree.root.children) == 0


def test_tree_with_unbalanced_classes():
    data = np.array([[0], [1], [1], [1], [1]])
    targets = np.array([0, 1, 1, 1, 1])
    tree = make_tree(data, targets)

    assert tree.predict(np.array([0])) == 0
    assert tree.predict(np.array([1])) == 1


def test_tree_recursive_split_structure():
    data = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    targets = np.array([0, 0, 1, 1])
    tree = make_tree(data, targets, features=[0, 1], max_depth=2)

    assert tree.root.feature in {0, 1}
    assert any(child.target is not None for child in tree.root.children.values())


def test_tree_handles_empty_dataset():
    data = np.empty((0, 1))
    targets = np.empty((0,))
    config = ID3Config(
        max_depth=1,
        eval_function=InformationGain(),
        tournament_size=2,
    )

    tree = ID3Tree(config=config)
    tree.fit(data=data, targets=targets, features=[0])

    assert tree.root.target is None or tree.root.target == tree.root.default_label


def test_tree_predict_on_empty_children_dict():
    tree = ID3Tree(config=ID3Config(1, InformationGain(), 2))
    tree.fit(
        data=np.array([[0]]),
        targets=np.array([1]),
        features=[0],
    )
    assert tree.predict(np.array([9999])) == 1


def test_tournament_selection_returns_valid_feature():
    data = np.array([[0], [1], [2], [3]])
    targets = np.array([0, 1, 0, 1])

    config = ID3Config(
        max_depth=3,
        eval_function=InformationGain(),
        tournament_size=4,
    )
    tree = ID3Tree(config=config)
    tree.fit(data=data, targets=targets, features=[0])

    assert tree.root.feature in {0}
