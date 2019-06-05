from __future__ import annotations
from typing import List
from typing import Optional
import numpy as np


class KeyPoint:
    def __init__(self, index: int, x: float, y: float, features):
        self.index = index
        self.x = x
        self.y = y
        self.features = features
        self.closest: Optional[KeyPoint] = None
        self.neighbours: List[KeyPoint] = []
        self.distance = -1

    def abs_distance(self, other: KeyPoint):
        return np.sum(np.abs(self.features - other.features))

    def get_closest(self, key_points: List[KeyPoint]):
        best_dist = float('inf')
        closest = None
        for keyPoint in key_points:
            distance = self.abs_distance(keyPoint)
            if distance < best_dist:
                best_dist = distance
                closest = keyPoint
        return closest

    def square_distance(self, other_key_point: KeyPoint) -> float:
        return (self.x - other_key_point.x) ** 2 + (self.y - other_key_point.y) ** 2

    def euclidean_distance(self, other_x: float, other_y: float):
        return np.sqrt((self.x - other_x) ** 2 + (self.y - other_y) ** 2)

    def calculate_neighbours(self, neighbourhood_size: int, all_key_points: List[KeyPoint]):
        for key_point in all_key_points:
            key_point.distance = key_point.square_distance(self)

        all_key_points.sort(key=lambda point: point.distance)
        found_neighbours = 0
        for key_point in all_key_points:
            if key_point.distance > 0:
                self.neighbours.append(key_point)
                found_neighbours += 1
                if found_neighbours >= neighbourhood_size:
                    break

    def has_neighbour(self, key_point: KeyPoint):
        return key_point in self.neighbours

    @staticmethod
    def from_string(key_point_index: int, text: str) -> KeyPoint:
        data = text.split()
        x = float(data[0])
        y = float(data[1])
        params = np.asarray(data[5:], dtype=np.int32)
        return KeyPoint(key_point_index, x, y, params)
