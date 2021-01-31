import math

from typing import List

Vector = List[float]

Matrix = List[Vector]


def cols(x: Matrix) -> int:
    return len(x[0])


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
