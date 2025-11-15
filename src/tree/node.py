from dataclasses import dataclass
from typing import TypeVar

L = TypeVar("L")


@dataclass
class Node[L]:
    feature: int | None = None
    target: L | None = None
    children: dict[int, "Node[L]"] | None = None
    default_label: L | None = None
