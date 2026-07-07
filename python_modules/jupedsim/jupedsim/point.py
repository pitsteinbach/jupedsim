# SPDX-License-Identifier: LGPL-3.0-or-later

import math
from typing import Tuple

class Point(tuple):
    """2D point represented as an immutable (x, y) tuple of floats."""

    def __new__(cls, x: "float | Tuple[float, float]", y: float = 0) -> "Point":
        if isinstance(x, tuple):
            return super().__new__(cls, (float(x[0]), float(x[1])))
        return super().__new__(cls, (float(x), float(y)))

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    def __add__(self, other: "Point") -> "Point":
        return Point(self[0] + other[0], self[1] + other[1])

    def __sub__(self, other: "Point") -> "Point":
        return Point(self[0] - other[0], self[1] - other[1])

    def __mul__(self, scalar: float) -> "Point":
        return Point(self[0] * scalar, self[1] * scalar)

    def __rmul__(self, scalar: float) -> "Point":
        return Point(self[0] * scalar, self[1] * scalar)

    def norm(self) -> float:
        return math.sqrt(self[0] ** 2 + self[1] ** 2)

    def normalize(self) -> "Point":
        n = self.norm()
        if n == 0.0:
            raise ValueError("Cannot normalize a zero-length vector")
        return Point(self[0] / n, self[1] / n)

    def __repr__(self) -> str:
        return f"Point({self[0]}, {self[1]})"

    def scalar_product(self, p2: Point):
        return self.x * p2.x + self.y * p2.y

    def rotate_90_deg(self):
        return Point(-self.y, self.x)
