import numpy as np


def change_2d_vector_to_homogeneous_vector(vector_2d):
    vector_homogeneous = np.array([vector_2d[0], vector_2d[1], 1])
    return vector_homogeneous


def change_homogeneous_vector_to_2d_vector(vector_homogeneous):
    # vector_2d = [vector_homogeneous[i] / vector_homogeneous[-1] for i in range(len(vector_homogeneous))]
    # vector_2d = np.array(vector_2d[:-1])
    vector_2d = np.array(vector_homogeneous[:-1])
    return vector_2d

