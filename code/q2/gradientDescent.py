from collections import deque
import time
import random
import numpy as np

def initialiseParameters(dimension):
    return np.array([0 for _ in range(dimension+1)])

def gradientDescent(X_values, Y_values, learning_rate, convergence_value, batch_size, moving_average_window_size, max_epochs, gradientFunc, lossFunc, plot_duration=0.2):
    
    params = initialiseParameters(len(X_values[0]))

    #convert to numpy arrays for faster execution
    X_array = np.array(X_values)
    Y_array = np.array(Y_values)
    
    #shuffle the data
    indices = np.arange(len(X_array))
    np.random.shuffle(indices)
    X_values_shuffled = X_array[indices]
    Y_values_shuffled = Y_array[indices]

    #start index for batch
    start_index = 0

    #queue to store moving average of loss values - using deque for better performance
    loss_history = deque(maxlen=moving_average_window_size)
    current_sum_of_moving_window = 0

    #setting up data collection for plotting
    data_for_plotting = []
    data_for_plotting.append(params.copy())
    time_last_recorded_data = time.time()
    number_of_epochs = 0
    number_of_iterations = 0

    new_loss = lossFunc(X_values_shuffled, Y_values_shuffled, params, start_index, start_index + batch_size - 1)

    while True:
        number_of_iterations += 1
        #calculate gradient and loss for current batch
        gradient = gradientFunc(X_values_shuffled, Y_values_shuffled, params, start_index, start_index + batch_size - 1)
        old_loss = new_loss
        #update start index for next batch
        start_index += batch_size
        if start_index >= len(X_values_shuffled):
            number_of_epochs += 1
            print(f"Epoch {number_of_epochs} completed")
            start_index = start_index % len(X_values_shuffled)

        #calculate loss for new parameters for next batch
        new_params = params - learning_rate * gradient
        new_loss = lossFunc(X_values_shuffled, Y_values_shuffled, new_params, start_index, start_index + batch_size - 1)

        #update moving average using deque
        loss_diff = (new_loss - old_loss)
        if len(loss_history) == moving_average_window_size:
            current_sum_of_moving_window -= loss_history[0]
        loss_history.append(loss_diff)
        current_sum_of_moving_window += loss_diff

        #record data for plotting every 0.2 seconds
        if time.time() - time_last_recorded_data > plot_duration:
            data_for_plotting.append(new_params.copy())
            time_last_recorded_data = time.time()

        #convergence check, if moving average is less than convergence value, and a certain number of iterations have passed, break
        if (len(loss_history) == moving_average_window_size and abs(current_sum_of_moving_window) < convergence_value) or (number_of_epochs >= max_epochs):
            break

        #print(f"Old Loss: {old_loss}, New Loss: {new_loss}, Moving Average Loss: {current_sum_of_moving_window / q.qsize()}")
        #update parameters
        params = new_params
    
    data_for_plotting.append(params.copy())
    return params, data_for_plotting, number_of_iterations