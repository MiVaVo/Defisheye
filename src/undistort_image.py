import cv2
import numpy as np

from src.utils import return_n_of_section


class UndistortImageOnLines:
    def __init__(self, line_min_threshold=340, angle_max_threshold=5):
        self.line_min_threshold = line_min_threshold
        self.angle_max_threshold = angle_max_threshold
        self.img_fish = None
        self.img_fish_lined = None
        self.img_undistorted = None
        self.img_undistorted_lines = None

    def configure_undistort_parameters(self, img):
        size = img.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        self.K = np.array([[focal_length, 0., center[0]],
                           [0., focal_length, center[1]],
                           [0., 0., 1.]])
        self.Knew = self.K.copy()
        self.Knew[(0, 1), (0, 1)] = 1 * self.Knew[(0, 1), (0, 1)]
        pass

    def undistort_image(self, imge):
        self.img_fish = imge
        self.configure_undistort_parameters(imge)
        lines, self.img_fish_lined = return_n_of_section(self.img_fish,
                                                         threshold=self.line_min_threshold,
                                                         angle_th=self.angle_max_threshold)
        lines_maks_list = []
        n_lines_max = 0
        j = 0
        all_lines_inf = []
        for i in np.linspace(-1, 0.2, 7):
            for h in np.linspace(-0.7, 0.2, 5):
                for k in np.linspace(-0.7, 0.2, 5):
                    D = np.array([i + 0.01, j + 0.01, h + 0.01, k + 0.01])
                    img_undistorted = cv2.fisheye.undistortImage(self.img_fish, self.K, D=D, Knew=self.Knew)
                    lines, img_lined_und = return_n_of_section(img_undistorted,
                                                               threshold=self.line_min_threshold,
                                                               angle_th=self.angle_max_threshold)
                    n_lines = lines.shape[0]
                    all_lines_inf.append([n_lines, [i, j, h, k]])
                    if n_lines >= n_lines_max:
                        n_lines_max = n_lines
                        lines_maks_list = [n_lines_max, [i, j, h, k]]

        i, j, h, k = lines_maks_list[1]
        D_found = np.array([i + 0.01, j + 0.01, h + 0.01, k + 0.01])
        self.img_undistorted = cv2.fisheye.undistortImage(self.img_fish,
                                                          self.K,
                                                          D=D_found,
                                                          Knew=self.Knew)
        lines, self.img_undistorted_lines = return_n_of_section(self.img_undistorted,
                                                                threshold=self.line_min_threshold)
        pass

    @staticmethod
    def draw_all_results(undistorted_image_class):
        cv2.imshow("Initial distorted (fisheyed) image", undistorted_image_class.img_fish)
        cv2.imshow("Initial image with found lines", undistorted_image_class.img_fish_lined)
        cv2.imshow("Undistorted image", undistorted_image_class.img_undistorted)
        cv2.imshow("Undistorted image with found lines", undistorted_image_class.img_undistorted_lines)
        cv2.waitKey(0);
        cv2.destroyAllWindows()
        pass

    @staticmethod
    def save_all_results(undistorted_image_class,path="result_imgs"):
        cv2.imwrite(path+"/"+"Initial_distorted_(fisheyed)_image.jpg", undistorted_image_class.img_fish)
        cv2.imwrite(path+"/"+"Initial_image_with_found_lines.jpg", undistorted_image_class.img_fish_lined)
        cv2.imwrite(path+"/"+"Undistorted_image.jpg", undistorted_image_class.img_undistorted)
        cv2.imwrite(path+"/"+"Undistorted_image_with_found_lines.jpg", undistorted_image_class.img_undistorted_lines)
        pass
