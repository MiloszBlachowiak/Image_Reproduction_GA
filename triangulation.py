import numpy as np
from scipy.spatial import Delaunay
from skimage import filters, morphology, color
import pandas as pd

# implementation of methods was based on the code from this github repository: https://github.com/ijmbarr/images-to-triangles

class Triangulation:

    def __init__(self, triangles=None) -> None:
        self.triangles = triangles

    @staticmethod
    def gaussian_mask(x, y, shape, amp=1, sigma=15):
        xv, yv = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        g = amp * np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * sigma ** 2))
        return g

        # get points along the four edges of the image
    def get_edge_points(self, image: np.ndarray, n_vertical=10, n_horizontal=10):

        vertical_len, horizontal_len = image.shape[:2]

        dx = int(horizontal_len / n_horizontal)
        dy = int(vertical_len / n_vertical)

        points = [(0, 0), (horizontal_len, 0), (0, vertical_len), (horizontal_len, vertical_len)] # vertices
        points += [(dx * i, 0) for i in range (1, n_horizontal)]
        points += [(dx * i, vertical_len) for i in range(1, n_horizontal)]
        points += [(0, dy * i) for i in range(1, n_vertical)]
        points += [(horizontal_len, dy * i) for i in range(1, n_vertical)]

        return np.array(points)

    # generate points across the image based on maximum entropy
    def get_points(self, image: np.ndarray, n_points: int=20, n_vertical=10, n_horizontal=10 ,
                   entropy_width=0.2, filter_width=0.2, suppression_width=0.2, suppression_amplitude=0.2):

        vertical_len, horizontal_len = image.shape[:2]

        length_scale = np.sqrt(horizontal_len * vertical_len / n_points)
        entropy_width = length_scale * entropy_width
        filter_width = length_scale * filter_width
        suppression_width = length_scale * suppression_width

        # convert to grayscale
        im2 = color.rgb2gray(image)

        # filter
        im2 = (
            255 * filters.gaussian(im2, sigma=filter_width)
        ).astype("uint8")

        # calculate entropy
        im2 = filters.rank.entropy(im2, morphology.disk(entropy_width))

        points = []
        for _ in range(n_points):
            y, x = np.unravel_index(np.argmax(im2), im2.shape)
            im2 -= self.gaussian_mask(x, y,
                                shape=im2.shape[:2],
                                amp=suppression_amplitude,
                                sigma=suppression_width)
            points.append((x, y))

        points = np.concatenate((points, self.get_edge_points(image, n_vertical, n_horizontal)))


        return points 

    def get_pixel_coords(self, img_shape_2D):
        # create a list of all pixel coordinates
        ymax, xmax = img_shape_2D
        xx, yy = np.meshgrid(np.arange(xmax), np.arange(ymax))
        pixel_coords = np.c_[xx.ravel(), yy.ravel()]
        return pixel_coords
    
    # for each pixel, identify which triangle it belongs to
    def get_simplex(self, image):
        pixel_coords = self.get_pixel_coords(image.shape[:2])
        return self.triangles.find_simplex(pixel_coords)

    # returns an array of size (n_triangles x 3) containing colours of the triangles
    def get_triangle_colour(self, image, agg_func=np.median):

        triangles_for_coord = self.get_simplex(image)

        df = pd.DataFrame({
            "triangle": triangles_for_coord,
            "r": image.reshape(-1, 3)[:, 0],
            "g": image.reshape(-1, 3)[:, 1],
            "b": image.reshape(-1, 3)[:, 2]
        })

        n_triangles = self.triangles.vertices.shape[0]

        by_triangle = (
            df
                .groupby("triangle")
            [["r", "g", "b"]]
                .aggregate(agg_func)
                .reindex(range(n_triangles), fill_value=0)
            # some triangles might not have pixels in them
        )

        return by_triangle.values.astype('int')

    # calculate how many pixels belong to each triangle
    def get_triangle_weights(self, image):
        triangles_for_coord = self.get_simplex(image)
        n_triangles = self.triangles.vertices.shape[0]
        return np.array([np.sum(triangles_for_coord == i) for i in range(n_triangles)])


    def triangulate(self, image, n_points=2000, n_vertical=50, n_horizontal=50):
        points = self.get_points(image, n_points, n_vertical, n_horizontal)
        self.triangles = Delaunay(points)  
