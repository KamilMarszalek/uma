from src.data.encoders import CatEncodingStrategy
from src.data.uci_data_provider import get_uci_data
from src.forest.config import TournamentForestConfig
from src.forest.forest import TournamentForest
from src.tree.config import TreeConfig
from src.tree.eval_func import EvalFunction
from src.tree.tree import TreeClass

TRAIN_SIZE = 0.7
RANDOM_SEED = 42


def main() -> None:
    train_data, test_data, train_targets, test_targets = get_uci_data(
        set_id=73,
        train_size=TRAIN_SIZE,
        random_seed=RANDOM_SEED,
        encode=CatEncodingStrategy.CATEGORICAL,
    )

    n_features = train_data.shape[1]
    print(f"Number of features: {n_features}")

    config = TournamentForestConfig(
        num_of_trees=15,
        sample_ratio=0.8,
        # feature_ratio=np.sqrt(data_np.shape[1]) / data_np.shape[1],
        feature_ratio=0.8,
        eval_function=EvalFunction.CART_GINI_GAIN,
        max_depth=10,
        tournament_size=4,
        tree_class=TreeClass.CART,
        tree_config_class=TreeConfig.CART,
    )

    forest = TournamentForest(train_data, train_targets, config)
    forest.build()

    correct = 0
    for x, y_true in zip(test_data, test_targets, strict=True):
        y_pred = forest.predict(x)
        if y_pred == y_true:
            correct += 1

    accuracy = correct / test_data.shape[0]
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
