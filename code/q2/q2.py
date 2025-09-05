import random
import numpy as np
import matplotlib.pyplot as plt
from gradientDescent import gradientDescent
from meanSquareErrorLossGradient import calculateGradientLinearRegression, calculateLossFunctionLinearRegression

random.seed(42)
np.random.seed(42)

#function to generate the data set
def generator(params, size):

    X_values = []
    Y_values = []

    for _ in range(size):

        x1 = np.random.normal(loc=3, scale=2)
        x2 = np.random.normal(loc=-1, scale=2)
        err = np.random.normal(loc=0, scale= np.sqrt(2))

        y = params[0] + params[1] * x1 + params[2] * x2 + err

        X_values.append([x1, x2])
        Y_values.append(y)

    return X_values, Y_values

#compute matrix closed form solution
def closed_form_solution(X_values, Y_values):

    m = len(X_values)
    #adding 1s for intercept
    X_b = np.column_stack([np.ones(m), X_values])
    Y_array = np.array(Y_values)

    #closed-form solution (Normal Equation): theta = (X^T * X)^(-1) * X^T * Y
    theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y_array
    return theta_best.tolist()


def main():

    X, Y = generator([3, 1, 2], 1000000)

    #splitting into training and test sets
    X_train = X[:800000]
    Y_train = Y[:800000]
    X_test = X[800000:]
    Y_test = Y[800000:]

    #calculating closed form solution and its losses
    closed_form_params = closed_form_solution(X_train, Y_train)
    closed_form_test_loss = calculateLossFunctionLinearRegression(X_test, Y_test, closed_form_params, 0, len(X_test)-1)
    closed_form_training_loss = calculateLossFunctionLinearRegression(X_train, Y_train, closed_form_params, 0, len(X_train)-1)

    #running gradient descent with different batch sizes and moving average window sizes and max number of epochs
    batch_sizes = [1, 80, 8000, 800000]
    window_sizes = [200000, 2000, 20, 1]
    number_of_epochs = [10, 100, 1000, 15000]
    plot_duration = [0.001, 0.005, 0.1, 0.2]
    colors =['red', 'blue', 'green', 'black']

    #set up to store results of the runs of SGD
    parameters = []
    test_losses = []
    training_losses = []
    iterations_to_converge = []
    
    #setting up 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    for i in range(len(batch_sizes)):

        #running gradient descent and calculating losses
        params, data_for_plotting, number_of_iterations = gradientDescent(X_train, Y_train, learning_rate=0.001, convergence_value=0.000001, batch_size=batch_sizes[i], moving_average_window_size=window_sizes[i], max_epochs=number_of_epochs[i], plot_duration = plot_duration[i], gradientFunc=calculateGradientLinearRegression, lossFunc=calculateLossFunctionLinearRegression)
        test_loss = calculateLossFunctionLinearRegression(X_test, Y_test, params, 0, len(X_test)-1)
        training_loss = calculateLossFunctionLinearRegression(X_train, Y_train, params, 0, len(X_train)-1)

        #storing parameters and losses
        parameters.append(params)
        test_losses.append(test_loss)
        training_losses.append(training_loss)
        iterations_to_converge.append(number_of_iterations)

        #plotting the path in 3D
        theta0_to_plot = [d[0] for d in data_for_plotting]
        theta1_to_plot = [d[1] for d in data_for_plotting]
        theta2_to_plot = [d[2] for d in data_for_plotting]
        ax.scatter(theta0_to_plot, theta1_to_plot, theta2_to_plot, color=colors[i], s=50, label=f"Batch size: {batch_sizes[i]}")
        ax.plot(theta0_to_plot, theta1_to_plot, theta2_to_plot, color=colors[i], linewidth=2)
    
    #printing results
    for i in range(len(batch_sizes)):
        print(f"Batch size: {batch_sizes[i]}, Parameters: {parameters[i]}, Training Loss: {training_losses[i]}, Test Loss: {test_losses[i]}, Iterations to Converge: {iterations_to_converge[i]}")
    print(f"Closed form solution parameters: {closed_form_params}, Training Loss: {closed_form_training_loss}, Test Loss: {closed_form_test_loss}")

    #finalizing 3D plot
    ax.set_xlabel("Theta 0")
    ax.set_ylabel("Theta 1")
    ax.set_zlabel("Theta 2")
    ax.set_title("3D Visualization of Gradient Descent")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()