import numpy as np
import cv2
from scipy.optimize import fsolve


class ImageRadialDistorter:
    '''
    Class used for synthetic image distortion.
    '''

    def __init__(self, k=0.00006, adjust_mapping_to_image_size=True):
        '''

        :param k: barrel distortion parameter
        :param image_shape: will be good if is knows
        :param adjust_mapping_to_image_size: if true, mapping for distortion will be reprojected to the same corner.
        '''
        self.k = k
        self.distortion_parameters_defined = False
        self.adjust_mapping_to_image_size = adjust_mapping_to_image_size
        self.image_shape = None

    def distort_image(self, img):
        if self.distortion_parameters_defined == False:
            self.define_distortion_parameters(img)
        img_dist = cv2.rotate(
            cv2.remap(img, self.map_x_undist_to_dist, self.map_y_undist_to_dist, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT), 0)
        img_dist = cv2.flip(img_dist, 1)
        # img_dist = cv2.resize(img_dist,None, fx=self.image_squeeze_factor,fy=self.image_squeeze_factor)
        return img_dist

    def distort_back_image(self, img_dist):
        '''
        Function to undistort image based on known mappings. Can not be used without prerviously distorting image.
        :param img_dist: distorted image
        :return: distorted back image based on known mappings.
        '''
        if self.distortion_parameters_defined == False:
            raise ValueError("Image needs to be distorted beforehand")
        x, y = np.mgrid[0:self.width:1, 0:self.height:1]
        mymatrix = np.concatenate([x[:, :, np.newaxis], y[:, :, np.newaxis]], axis=2)
        res_m = np.apply_along_axis(self.numpy_sol_finder, 2, mymatrix)
        self.map_x_dist_to_undist = res_m[:, :, 0].astype(np.float32)
        self.map_y_dist_to_undist = res_m[:, :, 1].astype(np.float32)

        return cv2.rotate(cv2.remap(img_dist, self.map_x_dist_to_undist, self.map_y_dist_to_undist,
                                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), 0)

    def define_distortion_parameters(self, image):
        if self.image_shape is None:
            self.height, self.width, self.channel = image.shape
        x, y = np.mgrid[0:self.width:1, 0:self.height:1]
        x = x.astype(np.float32) - self.width / 2
        y = y.astype(np.float32) - self.height / 2
        theta = np.arctan2(y, x)
        self.d = (x * x + y * y) ** 0.5
        self.r = self.d * (1 + self.k * self.d * self.d)
        # cv2.imshow("image",np.asarray((self.r-np.min(self.r))/(np.max(self.r)-np.min(self.r))*225).astype("uint8"));cv2.waitKey(0);cv2.destroyAllWindows()
        if self.adjust_mapping_to_image_size:
            self.r = self.d[0, 0] / self.r[0, 0] * self.r
        self.x_center = self.width / 2
        self.y_center = self.height / 2
        self.map_x_undist_to_dist = self.r * np.cos(theta) + self.width / 2
        self.map_y_undist_to_dist = self.r * np.sin(theta) + self.height / 2
        self.distortion_parameters_defined = True
        pass

    def solution_finder(self, x_new, y_new):
        data = (self.k, x_new, y_new, self.x_center, self.y_center)
        x, y = fsolve(ImageRadialDistorter.inverse_distorion, (1, 1), args=data)
        return x, y

    @staticmethod
    def inverse_distorion(p, *data):
        k, x_new, y_new, x_center, y_center = data
        x, y = p
        return (x_new - (1 + k * ((x - x_center) ** 2 + (y - y_center) ** 2)) * (x - x_center) - x_center,
                y_new - (1 + k * ((x - x_center) ** 2 + (y - y_center) ** 2)) * (y - y_center) - y_center)

    def numpy_sol_finder(self, intput):
        return self.solution_finder(x_new=intput[0],
                                    y_new=intput[1])

    @staticmethod
    def draw_image(img):
        random_id_of_img = str(np.random.randint(0, 10000, 1)[0])
        cv2.imshow("img_fish" + str(random_id_of_img), img);
        cv2.waitKey(0);
        cv2.destroyAllWindows()
        pass
