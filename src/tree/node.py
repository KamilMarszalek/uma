from dataclasses import dataclass
from typing import Any


@dataclass
class Node:
    feature: int | None = None
    target: Any | None = None
    children: dict[int, "Node"] | None = None
    default_label: Any | None = None
