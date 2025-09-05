import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def listFromFile(infile):
    dataframe = pd.read_csv(infile, header = None)
    values = dataframe.values.tolist()
    return values

def sigmoid(z):
    z = np.asarray(z)
    #preventing overflow while keeping vectorization
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def newtonsMethod(X_values, Y_values):

    #setting up initial parameters
    parameters = np.array([0.0 for _ in range(len(X_values[0]) + 1)])
    m = len(X_values)
    #appending 1s to deal with intercept using efficient NumPy operations
    X_with_ones = np.column_stack([np.ones(m), X_values])

    while True:
        
        #calculating predictions
        predictions = X_with_ones @ parameters
        predictions = sigmoid(predictions)

        #calculating matrix and gradient for update
        hessian_matrix = hessian(X_with_ones, parameters, predictions)
        gradient = gradientFunc(X_with_ones, Y_values, parameters, predictions)

        
        #updating parameters
        new_parameters = parameters - np.linalg.inv(hessian_matrix) @ gradient
        
        #checking for convergence
        if np.linalg.norm(new_parameters - parameters) < 1e-6:
            break

        parameters = new_parameters

    return parameters

#calculating hessian matrix for logistic regression
def hessian(X_with_ones, parameters, predictions):
    n = len(X_with_ones[0])
    m = len(X_with_ones)
    weights = predictions * (predictions - 1)
    hessian_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            hessian_matrix[i][j] = np.sum(X_with_ones[:, i] * X_with_ones[:, j] * weights)
    return hessian_matrix

#calculating gradient for logistic regression
def gradientFunc(X_with_ones, Y_values, parameters, predictions):
    m = len(X_with_ones)
    gradient_vector = np.zeros(len(X_with_ones[0]))
    for i in range(m):
        gradient_vector += (Y_values[i] - predictions[i]) * X_with_ones[i]
    return gradient_vector

def main():
    #extracting and normalizing data
    X_values = listFromFile("logisticX.csv")
    Y_values = listFromFile("logisticY.csv")
    X_values = np.array(X_values)
    Y_values = np.array(Y_values).flatten()
    X_values_normalized = (X_values - np.mean(X_values, axis=0)) / np.std(X_values, axis=0)

    #running newton's method
    parameters = newtonsMethod(X_values_normalized, Y_values)
    print(f"Parameters from Newton's Method: {parameters}")

    #plotting data points (using original, non-normalized data for visualization)
    X_values_zero_x  = [X_values[i][0] for i in range(len(Y_values)) if Y_values[i] == 0]
    X_values_zero_y  = [X_values[i][1] for i in range(len(Y_values)) if Y_values[i] == 0]
    X_values_one_x  = [X_values[i][0] for i in range(len(Y_values)) if Y_values[i] == 1]
    X_values_one_y  = [X_values[i][1] for i in range(len(Y_values)) if Y_values[i] == 1]

    #plotting all points
    plt.figure()
    plt.scatter(X_values_zero_x, X_values_zero_y, color='red', marker='x', label='Class 0')
    plt.scatter(X_values_one_x, X_values_one_y, color='blue', marker='o', label='Class 1')

    #finding min max values for graph
    min_x1 = min(X_values[:,0]) - 1
    max_x1 = max(X_values[:,0]) + 1
    min_x2 = min(X_values[:,1]) - 1
    max_x2 = max(X_values[:,1]) + 1

    #creating grid to find decision boundary points
    x1_range = np.linspace(min_x1, max_x1, 400)
    x2_range = np.linspace(min_x2, max_x2, 400)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]

    #finding points close to decision boundary
    boundary_points = []


    for point in grid_points:
        #normalize point before inputting into checker, as parameters are for normalized data
        normalized_point = (point - np.mean(X_values, axis=0)) / np.std(X_values, axis=0) 
        z = parameters[0] + parameters[1] * normalized_point[0] + parameters[2] * normalized_point[1]
        #checking if point is close to boundary
        if abs(z) < 0.01:
            boundary_points.append(point)

    boundary_points = np.array(boundary_points)
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], color='green', label='Decision Boundary')

    #extra plotting settings
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Data Points')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
