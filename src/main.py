from src.image_analyzer import ImageAnalyzer
from src.image_analyzer import RansacHeuristic
from src.image_analyzer import TransformationType

if __name__ == '__main__':
    image_analyzer: ImageAnalyzer = ImageAnalyzer("1.png", "2.png", 150, 0.9, 2000, 1, ransac_heuristic=RansacHeuristic.ITERATIONS,
                                                  transformation=TransformationType.PERSPECTIVE)
    image_analyzer.run()