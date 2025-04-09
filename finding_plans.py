import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import probabilistic_hough_line
from skimage.draw import line
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

def hough_transform(data):
    points_xy = data[['x', 'y']].values
    img_size = (1000, 1000)

    # Take the extremes of the points to normalize the coordinates
    x_min, x_max = points_xy[:, 0].min(), points_xy[:, 0].max()
    y_min, y_max = points_xy[:, 1].min(), points_xy[:, 1].max()

    # Normalization
    x_normalized = np.clip((points_xy[:, 0] - x_min) / (x_max - x_min) * (img_size[1] - 1), 0, img_size[1] - 1)
    y_normalized = np.clip((points_xy[:, 1] - y_min) / (y_max - y_min) * (img_size[0] - 1), 0, img_size[0] - 1)

    # Create a binary image
    edges = np.zeros(img_size, dtype=np.uint8)
    edges[y_normalized.astype(int), x_normalized.astype(int)] = 1

    # Hough Transform
    lines = probabilistic_hough_line(edges, threshold=10, line_length=10, line_gap=10)
    return lines, x_min, x_max, y_min, y_max

def calculate_slope_intercept(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    m = (y1 - y0) / (x1 - x0)
    b = y0 - m * x0
    return m, b

def cluster_slopes(lines):
    # Compute the slope and intercept for each line
    slopes = []
    intercepts = []
    for p0, p1 in lines:
        m, b = calculate_slope_intercept(p0, p1)
        slopes.append(m)
        intercepts.append(b)

    # stacking slopes and intercepts to create a feature matrix
    slopes = np.array(slopes)
    intercepts = np.array(intercepts)
    features = np.vstack((slopes, intercepts)).T

    # KMeans to cluster the lines
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(features)
    labels = kmeans.labels_

    # Fo each cluster, take the average slope and intercept
    equations = []
    for label in set(labels):
        cluster_indices = np.where(labels == label)[0]
        
        cluster_slopes = slopes[cluster_indices]
        cluster_intercepts = intercepts[cluster_indices]
        
        avg_slope = np.mean(cluster_slopes)
        avg_intercept = np.mean(cluster_intercepts)
        equations.append((avg_slope, avg_intercept))
        
        print(f"Cluster {label + 1}: y = {avg_slope:.2f}x + {avg_intercept:.2f}")
    return equations,labels

def denormalize_slope_intercept(slope_norm, intercept_norm, x_min, x_max, y_min, y_max):
    scale_x = x_max - x_min
    scale_y = y_max - y_min
    img_size = (1000, 1000)
    
    slope_denorm = slope_norm * scale_y / scale_x
    intercept_denorm = intercept_norm * scale_y/img_size[0] - slope_denorm * x_min +y_min
    
    return slope_denorm, intercept_denorm