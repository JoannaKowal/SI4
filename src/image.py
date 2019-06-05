from __future__ import annotations
from src.keypoint import KeyPoint
from typing import List
import numpy as np


class Image:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.features_path = file_name + ".haraff.sift"
        self.key_points: List[KeyPoint] = []
        self.features = None

    def import_key_points(self):
        file = open(self.features_path, mode="r")
        features_count: int = int(file.readline())
        key_points_count: int = int(file.readline())
        self.features = np.empty(shape=(key_points_count, features_count), dtype=np.int32)

        for key_point_index in range(key_points_count):
            line = file.readline()
            key_point: KeyPoint = KeyPoint.from_string(key_point_index, line)
            self.key_points.append(key_point)
            self.features[key_point_index] = key_point.features

    def calculate_closest_key_points(self, other_img: Image):
        n_key_points = len(self.key_points)
        for index in range(n_key_points):
            row = self.features[index, :]
            distances = np.sum(np.abs(row - other_img.features), axis=1)
            closest_index = np.argmin(distances)
            self.key_points[index].closest = other_img.key_points[closest_index]
