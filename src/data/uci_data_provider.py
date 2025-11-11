import pandas as pd
from ucimlrepo import fetch_ucirepo


# 73 is mushroom dataset
def get_uci_data(set_id: int = 73) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = fetch_ucirepo(id=set_id)
    X = dataset.data.features
    Y = dataset.data.targets
    return X, Y
