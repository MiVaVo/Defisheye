import cv2
from src.distort_image import ImageRadialDistorter
from src.undistort_image import UndistortImageOnLines

if __name__ == "__main__":
    file = "./imgs/not_fisheye_3.jpg"
    img_initial = cv2.imread(file)

    # distort normal image
    DistortClass = ImageRadialDistorter(k=0.000005)
    img_fish = DistortClass.distort_image(img_initial)
    # ImageRadialDistorter.draw_image(img_fish)

    # undistort image
    UnDistortClass = UndistortImageOnLines(line_min_threshold=100, angle_max_threshold=2)
    UnDistortClass.undistort_image(img_fish)
    UnDistortClass.draw_all_results(UnDistortClass)
