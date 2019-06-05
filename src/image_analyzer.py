
import random
import subprocess
from enum import Enum
from typing import Optional, Tuple, Callable
from typing import List

import cv2

from src.image import Image
from src.keypoint import KeyPoint
import numpy as np

KeyPointPair = Tuple[KeyPoint, KeyPoint]


class RansacHeuristic(Enum):
    NONE = 0
    ITERATIONS = 1
    DISTRIBUTION = 2
    DISTANCE = 3

class TransformationType(Enum):
    AFFINE = 0
    PERSPECTIVE = 1


class ImageAnalyzer:
    def __init__(self, first_img_name: str, second_img_name: str, neighbourhood_size: int,
                 consistency_threshold: float, iterations: int, ransac_threshold: float,
                 transformation: TransformationType = TransformationType.AFFINE,
                 ransac_heuristic: RansacHeuristic = RansacHeuristic.NONE,
                 iteration_heuristic_probability: float = 0.9):
        self.first_img_name = first_img_name
        self.second_img_name = second_img_name
        self.first_img: Optional[Image] = None
        self.second_img: Optional[Image] = None
        self.key_point_pairs: List[KeyPointPair] = []
        self.consistent_key_point_pairs: List[KeyPointPair] = []
        self.neighbourhood_size = neighbourhood_size
        self.consistency_threshold = consistency_threshold
        self.iterations = iterations
        self.ransac_threshold = ransac_threshold
        self.best_ransac_consensus: List[KeyPointPair] = []
        self.best_model: Optional[np.ndarray] = None
        self.ransac_heuristic = ransac_heuristic
        self.iteration_heuristic_probability = iteration_heuristic_probability
        self.transformation = transformation

    def run(self):
        print("Extracting")
        self.extract_files()
        print("Importing key points")
        self.import_key_points()
        print("Calculating pairs")
        self.calculate_key_point_pairs()
        print("Calculating neighbourhoods")
        self.calculate_neighbourhoods()
        print("Analyzing consistency")
        self.analyze_consistency()
        print("Running ransac")

        if self.ransac_heuristic == RansacHeuristic.ITERATIONS:
            self.estimate_ransac_iterations()

        if self.transformation == TransformationType.AFFINE:
            def get_sample(distribution):
                return self.get_random_sample(3, distribution)

            def transform_function(sample):
                return self.get_affine_transform(sample)
        else:
            def get_sample(distribution):
                return self.get_random_sample(4, distribution)

            def transform_function(sample):
                return self.get_perspective_transform(sample)

        self.ransac(get_sample, transform_function)
        print(len(self.key_point_pairs))
        print(len(self.consistent_key_point_pairs))
        print(len(self.best_ransac_consensus))
        self.show_images()

    def extract_files(self):
        self.extract_image(self.first_img_name)
        self.extract_image(self.second_img_name)

    def import_key_points(self):
        self.first_img = Image(self.first_img_name)
        self.first_img.import_key_points()
        self.second_img = Image(self.second_img_name)
        self.second_img.import_key_points()

    def calculate_key_point_pairs(self):
        self.first_img.calculate_closest_key_points(self.second_img)
        self.second_img.calculate_closest_key_points(self.first_img)
        for key_point in self.first_img.key_points:
            if key_point.closest.closest == key_point:
                self.key_point_pairs.append((key_point, key_point.closest))

    def analyze_consistency(self):
        pair_number_threshold = int(self.neighbourhood_size * self.consistency_threshold)
        for pair_one in self.key_point_pairs:
            neighbour_pairs_count = 0
            for pair_two in self.key_point_pairs:
                if pair_one[0].has_neighbour(pair_two[0]) and pair_one[1].has_neighbour(pair_two[1]):
                    neighbour_pairs_count += 1
                    if neighbour_pairs_count >= pair_number_threshold:
                        self.consistent_key_point_pairs.append(pair_one)
                        break

    def calculate_neighbourhoods(self):
        first_img_paired_key_points = [pair[0] for pair in self.key_point_pairs]
        second_img_paired_key_points = [pair[1] for pair in self.key_point_pairs]

        for pair in self.key_point_pairs:
            pair[0].calculate_neighbours(self.neighbourhood_size, first_img_paired_key_points)
            pair[1].calculate_neighbours(self.neighbourhood_size, second_img_paired_key_points)

    def show_images(self):
        image1 = cv2.imread(self.first_img_name)
        image2 = cv2.imread(self.second_img_name)
        stacked_image = np.vstack((image1, image2))
        offset = image1.shape[0]
        for pair in self.key_point_pairs:
            point1 = pair[0]
            point2 = pair[1]
            cv2.line(stacked_image, (int(point1.x), int(point1.y)), (int(point2.x), int(point2.y + offset)),
                     (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1, cv2.LINE_AA)

        cv2.namedWindow("All pairs", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("All pairs", stacked_image.shape[1], 650)
        cv2.imshow("All pairs", stacked_image)

        cv2.waitKey(0)

        stacked_image2 = np.vstack((image1, image2))
        for pair in self.consistent_key_point_pairs:
            point1 = pair[0]
            point2 = pair[1]
            cv2.line(stacked_image2, (int(point1.x), int(point1.y)), (int(point2.x), int(point2.y + offset)),
                     (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1, cv2.LINE_AA)

        cv2.namedWindow("Consistent pairs", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Consistent pairs", stacked_image2.shape[1], 650)
        cv2.imshow("Consistent pairs", stacked_image2)

        cv2.waitKey(0)

        stacked_image2 = np.vstack((image1, image2))
        for pair in self.best_ransac_consensus:
            point1 = pair[0]
            point2 = pair[1]
            cv2.line(stacked_image2, (int(point1.x), int(point1.y)), (int(point2.x), int(point2.y + offset)),
                     (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1, cv2.LINE_AA)

        cv2.namedWindow("Transformed ransac pairs", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Transformed ransac pairs", stacked_image2.shape[1], 650)
        cv2.imshow("Transformed ransac pairs", stacked_image2)

        cv2.waitKey(0)

        if self.best_model is not None:
            if self.best_model[2][0] == 0 and self.best_model[2][1] == 0:
                image1 = cv2.warpAffine(image1, self.best_model[:2], (image1.shape[1], image1.shape[0]))
            else:
                image1 = cv2.warpPerspective(image1, self.best_model, (image1.shape[1], image1.shape[0]))

        cv2.namedWindow("Transformation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Transformation", image1.shape[1], image1.shape[0])
        cv2.imshow("Transformation", image1)
        cv2.waitKey(0)

    def ransac(self, get_sample: Callable[[List[KeyPointPair]], List[KeyPointPair]], get_transformation: Callable[[List[KeyPointPair]], np.ndarray]):
        best_consensus: List[KeyPointPair] = []
        best_transformation = None

        distribution = list(self.key_point_pairs)

        for i in range(self.iterations):
            current_consensus = []
            sample = get_sample(distribution)
            A = get_transformation(sample)
            for pair in self.key_point_pairs:
                point = [[pair[0].x], [pair[0].y], [1]]
                transformed_point = A @ np.array(point)
                transformed_point = transformed_point / transformed_point[2]
                distance = pair[1].euclidean_distance(transformed_point[0], transformed_point[1])
                if distance < self.ransac_threshold:
                    current_consensus.append(pair)

            if len(current_consensus) > len(best_consensus):
                best_consensus = current_consensus
                best_transformation = A
                if self.ransac_heuristic == RansacHeuristic.DISTRIBUTION:
                    distribution.extend(sample)
        self.best_ransac_consensus = best_consensus
        self.best_model = best_transformation

    def get_affine_transform(self, pairs: List[KeyPointPair]):
        pair1 = pairs[0]
        pair2 = pairs[1]
        pair3 = pairs[2]

        B = np.array(
            [
                [pair1[0].x, pair1[0].y, 1, 0, 0, 0],
                [pair2[0].x, pair2[0].y, 1, 0, 0, 0],
                [pair3[0].x, pair3[0].y, 1, 0, 0, 0],
                [0, 0, 0, pair1[0].x, pair1[0].y, 1],
                [0, 0, 0, pair2[0].x, pair2[0].y, 1],
                [0, 0, 0, pair3[0].x, pair3[0].y, 1]
            ]
        )

        C = np.array(
            [
                [pair1[1].x],
                [pair2[1].x],
                [pair3[1].x],
                [pair1[1].y],
                [pair2[1].y],
                [pair3[1].y]
            ]
        )

        result = np.linalg.inv(B) @ C
        return np.reshape(np.append(result, [[0], [0], [1]]), (3, 3))

    def get_perspective_transform(self, pairs: List[KeyPointPair]):
        pair1 = pairs[0]
        pair2 = pairs[1]
        pair3 = pairs[2]
        pair4 = pairs[3]

        B = np.array(
            [
                [pair1[0].x, pair1[0].y, 1, 0, 0, 0, -pair1[1].x * pair1[0].x, -pair1[1].x * pair1[0].y],
                [pair2[0].x, pair2[0].y, 1, 0, 0, 0, -pair2[1].x * pair2[0].x, -pair2[1].x * pair2[0].y],
                [pair3[0].x, pair3[0].y, 1, 0, 0, 0, -pair3[1].x * pair3[0].x, -pair3[1].x * pair3[0].y],
                [pair4[0].x, pair4[0].y, 1, 0, 0, 0, -pair4[1].x * pair4[0].x, -pair4[1].x * pair4[0].y],
                [0, 0, 0, pair1[0].x, pair1[0].y, 1, -pair1[1].y * pair1[0].x, -pair1[1].y * pair1[0].y],
                [0, 0, 0, pair2[0].x, pair2[0].y, 1, -pair2[1].y * pair2[0].x, -pair2[1].y * pair2[0].y],
                [0, 0, 0, pair3[0].x, pair3[0].y, 1, -pair3[1].y * pair3[0].x, -pair3[1].y * pair3[0].y],
                [0, 0, 0, pair4[0].x, pair4[0].y, 1, -pair4[1].y * pair4[0].x, -pair4[1].y * pair4[0].y]
            ]
        )

        C = np.array(
            [
                [pair1[1].x],
                [pair2[1].x],
                [pair3[1].x],
                [pair4[1].x],
                [pair1[1].y],
                [pair2[1].y],
                [pair3[1].y],
                [pair4[1].y]
            ]
        )

        result = np.linalg.inv(B) @ C
        return np.reshape(np.append(result, [[1]]), (3, 3))

    def get_random_sample(self, number_of_points: int, key_point_pairs: List[KeyPointPair]):
        result: List[KeyPointPair] = []
        while len(result) < number_of_points:
            random_pair = random.choice(key_point_pairs)
            pair_correct = True
            for pair in result:
                if (pair[0].x == random_pair[0].x and pair[0].y == random_pair[0].y) or \
                        (pair[1].x == random_pair[1].x and pair[1].y == random_pair[1].y):
                    pair_correct = False
                    break
            if pair_correct:
                result.append(random_pair)

        return result

    def estimate_ransac_iterations(self):
        if len(self.key_point_pairs) == 0:
            return 0
        else:
            n = 3 if self.transformation == TransformationType.AFFINE else 4
            w = len(self.consistent_key_point_pairs) / len(self.key_point_pairs)
            self.iterations = int(np.log2(1 - self.iteration_heuristic_probability) / np.log2(1 - w ** n))
            print(f"estimated: {self.iterations}")

    @staticmethod
    def extract_image(image_path: str):
        subprocess.call(["extract_features_32bit.exe", '-haraff', '-sift', '-i', image_path, '-DE'])
