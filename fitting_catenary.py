import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cluster_point_per_plan(data,denormalized_equations):
    points_xyz = data[['x', 'y', 'z']].values
    x_all, y_all, z_all = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]


    cluster_points_list = [[] for _ in range(len(denormalized_equations))]

    for i in range(len(points_xyz)):
        x, y = x_all[i], y_all[i]
        
        # compute the distance of each point to the predicted lines
        distances = []
        for m, b in denormalized_equations:
            d = np.abs(m * x - y + b) / np.sqrt(m ** 2 + 1)
            distances.append(d)
        
        # associate each point to the cluster with the minimum distance
        best_cluster = np.argmin(distances)
        cluster_points_list[best_cluster].append(points_xyz[i])

    cluster_points_list = [np.array(points) for points in cluster_points_list]
    return cluster_points_list

def catenary_equation(x, y0, c, x0):
    return y0 + c * (np.cosh((x - x0) / c) - 1)

def get_u_and_z(points_xyz, slope, intercept):
    # Line direction in (x, y)
    direction = np.array([1, slope])
    direction /= np.linalg.norm(direction)

    # Project each (x, y) point onto the direction vector
    xy = points_xyz[:, :2]
    projection_lengths = np.dot(xy, direction)
    
    # Recover z
    z = points_xyz[:, 2]
    
    return projection_lengths, z