import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gradientDescent import gradientDescent
from meanSquareErrorLossGradient import calculateGradientLinearRegression, calculateLossFunctionLinearRegression


def listFromCSV(infile):
    dataframe = pd.read_csv(infile, header = None)
    values = dataframe.values.tolist()
    return values

# q1. part 2 plotting function
def plotValuesAndLine(X_values, Y_values, params):    
    plt.scatter(X_values, Y_values, color='blue', label='Data points')
    x_range = np.linspace(min(X_values)[0], max(X_values)[0], 100)
    y_range = params[0] + params[1] * x_range
    plt.plot(x_range, y_range, color='red', label='Regression line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig("LinearRegressionPlot.png")
    plt.close()

#q1. part 3 plotting function
def plotLossOverTime(X_values, Y_values, data_for_plotting):
    
    #data for surface plot
    theta0_grid = np.linspace(-25, 35, 100)
    theta1_grid = np.linspace(0, 60, 100)
    X, Y = np.meshgrid(theta0_grid, theta1_grid)
    Z = np.array([[calculateLossFunctionLinearRegression(X_values, Y_values, [X[i,j], Y[i,j]], 0, len(X_values)) for j in range(len(X[0]))] for i in range(len(X))])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #surface plot
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    
    #plot trajectory
    x_pts = [p[0] for p in data_for_plotting]
    y_pts = [p[1] for p in data_for_plotting]
    z_pts = [calculateLossFunctionLinearRegression(X_values, Y_values, p, 0, len(X_values)) for p in data_for_plotting]
    ax.scatter(x_pts, y_pts, z_pts, color='red', s=50, label="Gradient Descent Path")
    ax.plot(x_pts, y_pts, z_pts, color='red', linewidth=2)

    # naming axes, graph
    ax.set_xlabel('Theta 0 (Intercept)')
    ax.set_ylabel('Theta 1 (Slope)')
    ax.set_zlabel('Loss')
    plt.title('Loss Surface with Gradient Descent Path')
    plt.savefig("LossOverTime.png")
    plt.close()

# q1. part 4,5 plotting function
def runAndPlotContours(X_values, Y_values, learning_rate):
    #runnning with different learning rates
    _, data_for_plotting, number_of_epochs = gradientDescent(X_values, Y_values, learning_rate, convergence_value=0.000000001, batch_size=1000, moving_average_window_size=1, max_epochs = 1e9, gradientFunc=calculateGradientLinearRegression, lossFunc=calculateLossFunctionLinearRegression)

    print(f"For learning rate {learning_rate}, number of epochs: {number_of_epochs}")
    #collecting data for contour plot
    theta0_grid = np.linspace(-25, 35, 100)
    theta1_grid = np.linspace(0, 60, 100)
    X, Y = np.meshgrid(theta0_grid, theta1_grid)
    Z = np.array([[calculateLossFunctionLinearRegression(X_values, Y_values, [X[i,j], Y[i,j]], 0, len(X_values)) for j in range(len(X[0]))] for i in range(len(X))])

    #points where we have captured parameters
    theta0_to_plot = [d[0] for d in data_for_plotting]
    theta1_to_plot = [d[1] for d in data_for_plotting]
    losses = [calculateLossFunctionLinearRegression(X_values, Y_values, d, 0, len(X_values)) for d in data_for_plotting]

    #plotting contours for each captured parameter
    for loss in losses:
        plt.contour(X, Y, Z, levels=[loss], colors='grey', linewidths = 1)

    #plotting other contours
    plt.contour(X, Y, Z, levels=15, colors='blue', linewidths = 1)

    #plotting the gradient descent path
    plt.plot(theta0_to_plot, theta1_to_plot, color='red', marker='o', markersize=5, label='Gradient Descent Path')

    #plotting the final parameters
    plt.xlabel('Theta 0 (Intercept)')
    plt.ylabel('Theta 1 (Slope)')
    plt.title('Contour Plot of Loss Function at learning rate '+str(learning_rate))
    plt.legend()
    plt.savefig("ContourPlot"+str(learning_rate)+".png")
    plt.close()


def main():
    
    X_values = listFromCSV("linearX.csv")
    Y_values = listFromCSV("linearY.csv")
    Y_values = [y[0] for y in Y_values]

    #running gradient descent
    params, data_for_plotting, number_of_epochs = gradientDescent(X_values, Y_values, learning_rate=0.01, convergence_value=0.000000001, batch_size=1000, max_epochs = 1e9, moving_average_window_size=1, gradientFunc=calculateGradientLinearRegression, lossFunc=calculateLossFunctionLinearRegression)
    print(f"params: {params}, number_of_epochs: {number_of_epochs}")

    #plotting the results
    plotValuesAndLine(X_values, Y_values, params)
    plotLossOverTime(X_values, Y_values, data_for_plotting)
    learning_rates = [0.01, 0.001, 0.025, 0.1]
    for lr in learning_rates:
        runAndPlotContours(X_values, Y_values, lr)

    

if __name__ == "__main__":
    main()