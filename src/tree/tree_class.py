from enum import Enum
from typing import Any

from src.tree.cart_tree import CARTTree
from src.tree.id3_tree import ID3Tree
from src.tree.sklearn_tree import SklearnTreeWrapper


class TreeClass(Enum):
    ID3 = ID3Tree
    CART = CARTTree
    SKLEARN = SklearnTreeWrapper

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.value(*args, **kwargs)
