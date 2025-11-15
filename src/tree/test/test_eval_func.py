import numpy as np

from src.tree.eval_func import GainRatio, GiniGain, InformationGain

# -------------------------------
# INFORMATION GAIN
# -------------------------------


def test_information_gain_zero_when_no_split():
    data = np.array([[1], [1], [1]])
    targets = np.array([0, 0, 0])

    ig = InformationGain()
    assert ig(data, targets, 0) == 0.0


def test_information_gain_perfect_split_two_classes():
    data = np.array([[0], [0], [1], [1]])
    targets = np.array([0, 0, 1, 1])

    ig = InformationGain()
    # Entropy(full) = 1.0
    # Weighted entropy = 0
    assert ig(data, targets, 0) == 1.0


def test_information_gain_half_split_mixed_targets():
    data = np.array([[0], [0], [1], [1]])
    targets = np.array([0, 1, 0, 1])

    ig = InformationGain()
    # Entropy(full)=1.0, weighted entropy=1.0 → IG=0
    assert ig(data, targets, 0) == 0.0


def test_information_gain_real_split_three_groups():
    data = np.array([[0], [0], [1], [1], [1], [2], [2]])
    targets = np.array(
        [
            0,
            0,
            1,
            1,
            1,
            0,
            0,
        ]
    )

    ig = InformationGain()
    value = ig(data, targets, 0)
    assert value > 0.0
    assert value < 1.0


# -------------------------------
# GINI GAIN
# -------------------------------


def test_gini_gain_zero_when_no_split():
    data = np.array([[1], [1], [1]])
    targets = np.array([0, 0, 0])

    gg = GiniGain()
    assert gg(data, targets, 0) == 0.0


def test_gini_gain_perfect_split():
    data = np.array([[0], [0], [1], [1]])
    targets = np.array([0, 0, 1, 1])

    gg = GiniGain()
    # gini(full)=0.5, gini(left)=0.0, gini(right)=0.0 → gain=0.5
    assert gg(data, targets, 0) == 0.5


def test_gini_gain_no_information():
    data = np.array([[0], [0], [1], [1]])
    targets = np.array([0, 1, 0, 1])

    gg = GiniGain()
    # gini(full)=0.5, each split half has gini=0.5 → gain=0.0
    assert gg(data, targets, 0) == 0.0


def test_gini_gain_real_split_three_groups():
    data = np.array([[0], [0], [1], [1], [1], [2], [2]])
    targets = np.array([0, 0, 1, 1, 1, 0, 0])

    gg = GiniGain()
    value = gg(data, targets, 0)
    assert value > 0.0
    assert value < 0.5


# -------------------------------
# GAIN RATIO
# -------------------------------


def test_gain_ratio_zero_when_no_split_info():
    data = np.array([[1], [1], [1]])
    targets = np.array([0, 0, 1])

    gr = GainRatio()
    assert gr(data, targets, 0) == 0.0


def test_gain_ratio_perfect_split():
    data = np.array([[0], [0], [1], [1]])
    targets = np.array([0, 0, 1, 1])

    gr = GainRatio()
    value = gr(data, targets, 0)

    # Gain = 1.0, split_info = 1.0 → GR = 1.0
    assert value == 1.0


def test_gain_ratio_penalizes_many_value_splits():
    data_A = np.array([[0], [1], [2], [3]])
    targets = np.array([0, 0, 1, 1])

    data_B = np.array([[0], [0], [1], [1]])

    gr = GainRatio()

    gr_A = gr(data_A, targets, 0)
    gr_B = gr(data_B, targets, 0)

    assert gr_B > gr_A


def test_gain_ratio_general_case():
    data = np.array([[0], [0], [1], [1], [1], [2], [2]])
    targets = np.array([0, 0, 1, 1, 1, 0, 0])

    gr = GainRatio()
    value = gr(data, targets, 0)

    assert value > 0.0
    assert value < 1.0
