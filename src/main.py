from src.image_analyzer import ImageAnalyzer
from src.image_analyzer import RansacHeuristic
from src.image_analyzer import TransformationType

if __name__ == '__main__':
    image_analyzer: ImageAnalyzer = ImageAnalyzer("1.png", "2.png", 150, 0.9, 100, 1,
                                                  ransac_heuristic=RansacHeuristic.DISTRIBUTION,
                                                  transformation=TransformationType.PERSPECTIVE)
    image_analyzer.run()