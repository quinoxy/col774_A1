import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

#class as numbers for easier handling
class Label(Enum):
    ALASKA = 0
    CANADA = 1

#extracting points
def listFromFile(infile):
    dataframe = pd.read_csv(infile, header = None)
    values = dataframe.values.tolist()
    return values

#computes (x-mu)^T * Sigma^-1 * (x-mu)
def computeExponent(point, mean, covariance_matrix):
    diff = point - mean
    inv_cov = np.linalg.inv(covariance_matrix)
    return (diff.T @ inv_cov @ diff)

#computes if a point satisfies GDA boundary condition
def satisfiesGDABound(point,mean_zero, mean_one, covariance_matrix_zero, covariance_matrix_one):

    #auxilaries for computation
    covariance_zero_det = np.linalg.det(covariance_matrix_zero)
    covariance_one_det = np.linalg.det(covariance_matrix_one)
    exponent_zero = computeExponent(point, mean_zero, covariance_matrix_zero)
    exponent_one = computeExponent(point, mean_one, covariance_matrix_one)

    #according to formula given in report
    value_to_be_checked = np.log(covariance_zero_det / covariance_one_det) + exponent_one - exponent_zero

    #if value is close to 0, we consider it to be on the boundary
    if abs(value_to_be_checked) < 5e-2:
        return True
    return False

def linearDiscriminantAnalysis(X0_values, X1_values):

    m = len(X0_values) + len(X1_values)

    #compute means
    mean_zero = np.mean(X0_values, axis=0)
    mean_one = np.mean(X1_values, axis=0)

    #compute covariance matrix
    X0_temp = X0_values - mean_zero
    X1_temp = X1_values - mean_one
    covariance_matrix = (X0_temp.T @ X0_temp + X1_temp.T @ X1_temp) / m

    #printing parameters
    print("LDA parameters:")
    print(f"Mean Alaska: {mean_zero}")
    print(f"Mean Canada: {mean_one}")
    print(f"Covariance Matrix:\n {covariance_matrix}")

    return mean_zero, mean_one, covariance_matrix

def quadraticDiscriminantAnalysis(X0_values, X1_values):

    m = len(X0_values) + len(X1_values)

    #compute means
    mean_zero = np.mean(X0_values, axis=0)
    mean_one = np.mean(X1_values, axis=0)

    #compute covariance matrix
    X0_temp = X0_values - mean_zero
    X1_temp = X1_values - mean_one
    covariance_matrix_zero = (X0_temp.T @ X0_temp) / len(X0_values)
    covariance_matrix_one = (X1_temp.T @ X1_temp) / len(X1_values)
    print("QDA parameters:")
    print(f"Mean Alaska: {mean_zero}")
    print(f"Mean Canada: {mean_one}")
    print(f"Covariance Matrix Alaska:\n {covariance_matrix_zero}")
    print(f"Covariance Matrix Canada:\n {covariance_matrix_one}")

    return mean_zero, mean_one, covariance_matrix_zero, covariance_matrix_one

def main():

    #load  and clean dataset
    X_values = listFromFile('q4x.dat')
    Y_values = listFromFile('q4y.dat')
    X_values = np.array([[int(i) for i in x[0].split()] for x in X_values])
    X_values_normalized = (np.array(X_values) - np.mean(X_values, axis=0)) / np.std(X_values, axis=0)
    Y_values = np.array([Label.ALASKA if (y == ['Alaska']) else Label.CANADA for y in Y_values])
    X_values_alaska = np.array([X_values_normalized[i] for i in range(len(Y_values)) if Y_values[i] == Label.ALASKA])
    X_values_canada = np.array([X_values_normalized[i] for i in range(len(Y_values)) if Y_values[i] == Label.CANADA])

    #perform GDA
    linear_mean_alaska, linear_mean_canada, linear_covariance = linearDiscriminantAnalysis(X_values_alaska, X_values_canada)
    quad_mean_alaska, quad_mean_canada, quad_covariance_alaska, quad_covariance_canada = quadraticDiscriminantAnalysis(X_values_alaska, X_values_canada)
    
    #points for plotting (we need unnormalized points here)
    X_values_alaska_axis0 = [X_values[i][0] for i in range(len(X_values)) if Y_values[i] == Label.ALASKA]
    X_values_alaska_axis1 = [X_values[i][1] for i in range(len(X_values)) if Y_values[i] == Label.ALASKA]
    X_values_canada_axis0 = [X_values[i][0] for i in range(len(X_values)) if Y_values[i] == Label.CANADA]
    X_values_canada_axis1 = [X_values[i][1] for i in range(len(X_values)) if Y_values[i] == Label.CANADA]

    #finding min max values for graph
    min_x1 = min(X_values[:,0]) - 1
    max_x1 = max(X_values[:,0]) + 1
    min_x2 = min(X_values[:,1]) - 1
    max_x2 = max(X_values[:,1]) + 1

    #setting up plot
    plt.figure()

    #setting up points for plotting decision boundary
    x1_range = np.linspace(min_x1, max_x1, 400)
    x2_range = np.linspace(min_x2, max_x2, 400)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]

    #checking which points satisfy boundary condition
    linear_boundary_points = []
    quad_boundary_points = []

    for point in grid_points:

        #normalize point before inputting into checker, as means, covariances are for normalized data
        normalized_point = (point - np.mean(X_values, axis=0)) / np.std(X_values, axis=0)

        #boundary checks
        if satisfiesGDABound(normalized_point, linear_mean_alaska, linear_mean_canada, linear_covariance, linear_covariance):
            linear_boundary_points.append(point)
        if satisfiesGDABound(normalized_point, quad_mean_alaska, quad_mean_canada, quad_covariance_alaska, quad_covariance_canada):
            quad_boundary_points.append(point)
    
    linear_boundary_points = np.array(linear_boundary_points)
    quad_boundary_points = np.array(quad_boundary_points)

    #plotting boundary points
    if len(linear_boundary_points) > 0:
        plt.plot(linear_boundary_points[:,0], linear_boundary_points[:,1], color='green', label='LDA Boundary')
    if len(quad_boundary_points) > 0:
        plt.plot(quad_boundary_points[:,0], quad_boundary_points[:,1], color='orange', label='QDA Boundary')

    #setting up plot
    plt.scatter(X_values_alaska_axis0, X_values_alaska_axis1, color='blue', marker='o', label='Alaska')
    plt.scatter(X_values_canada_axis0, X_values_canada_axis1, color='red', marker='x', label='Canada')

    #extra plot settings
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('GDA: Data points and decision boundaries')
    plt.legend()
    plt.xlim(min_x1, max_x1)
    plt.ylim(min_x2, max_x2)
    plt.show()


if __name__ == "__main__":
    main()
