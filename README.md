# LiDAR Cable Detection and Modeling

## Summary of the Algorithm

This project aims to detect and model overhead cables from a LiDAR point cloud using geometric and mathematical techniques. The main steps of the algorithm are as follows:

### 1. LiDAR Data Visualization  
The point cloud is first visualized in 3D to provide an overview of the scene and locate potential cables.

### 2. Line Detection via Hough Transform  
A Hough transform is applied to the XY projection of the data to detect linear segments corresponding to cable projections on the ground plane.

### 3. Line Clustering  
Detected lines are grouped into clusters based on the similarity of their slopes. Each cluster is assumed to represent a distinct cable direction.

### 4. Line Equation Extraction  
For each cluster, the average line is computed to represent the underlying cable's 2D projection. These equations are then denormalized to match the original coordinate system.

### 5. Point Assignment to Lines  
The 3D points are assigned to the nearest fitted line (cluster) to isolate the points corresponding to each cable.

### 6. Coordinate Projection (u, z space)  
For each group of points, coordinates are projected onto the plane orthogonal to the fitted line, resulting in a (u, z) representation — where `u` is the arc-length along the cable direction and `z` is the height.

### 7. Catenary Curve Fitting  
A catenary model is fitted to each (u, z) cluster using nonlinear regression (curve_fit) and the equation of a 2D catenary.

### 8. Result Visualization  
Each cluster of points is visualized in the (u, z) space alongside the fitted catenary curve to verify the quality of the fit.


