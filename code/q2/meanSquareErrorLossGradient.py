import numpy as np

def calculateLossFunctionLinearRegression(X_values, Y_values, parameters, start_index, end_index):
    #k is batch size, n is dimension of space of X values
    m = len(X_values)
    k = end_index - start_index + 1
    
    # Use efficient NumPy slicing instead of list comprehension + np.array creation
    if hasattr(X_values, 'shape'):  # Already a NumPy array
        # Handle wraparound with modulo indexing for efficiency
        if start_index + k <= m:
            X_batch = X_values[start_index:end_index+1]
            Y_batch = Y_values[start_index:end_index+1]
        else:
            # Handle wraparound case
            indices = np.array([i % m for i in range(start_index, end_index+1)])
            X_batch = X_values[indices]
            Y_batch = Y_values[indices]
    else:
        # Fallback for list input (convert once)
        X_batch = np.array([X_values[i % m] for i in range(start_index, end_index+1)])
        Y_batch = np.array([Y_values[i % m] for i in range(start_index, end_index+1)])

    #appending 1s to deal with intercept using efficient NumPy operations
    X_batch_with_ones = np.column_stack([np.ones(k), X_batch])
    if not isinstance(parameters, np.ndarray):
        parameters = np.array(parameters)
    #calculating theta^T * X
    predictions = X_batch_with_ones @ parameters

    #calculating final loss
    loss = np.sum((predictions - Y_batch) ** 2) / (2 * k)
    return loss

def calculateGradientLinearRegression(X_values, Y_values, parameters, start_index, end_index):

    #k is batch size, n is dimension of space of X values
    m = len(X_values)
    k = end_index - start_index + 1
    
    # Use efficient NumPy slicing instead of list comprehension + np.array creation
    if hasattr(X_values, 'shape'):  # Already a NumPy array
        # Handle wraparound with modulo indexing for efficiency
        if start_index + k <= m:
            X_batch = X_values[start_index:end_index+1]
            Y_batch = Y_values[start_index:end_index+1]
        else:
            # Handle wraparound case
            indices = np.array([i % m for i in range(start_index, end_index+1)])
            X_batch = X_values[indices]
            Y_batch = Y_values[indices]
    else:
        # Fallback for list input (convert once)
        X_batch = np.array([X_values[i % m] for i in range(start_index, end_index+1)])
        Y_batch = np.array([Y_values[i % m] for i in range(start_index, end_index+1)])

    #appending 1s to deal with intercept using efficient NumPy operations
    X_batch_with_ones = np.column_stack([np.ones(k), X_batch])
    if not isinstance(parameters, np.ndarray):
        parameters = np.array(parameters)

    #calculating theta^T * X
    predictions = X_batch_with_ones @ parameters

    errors = predictions - Y_batch
    gradient = (X_batch_with_ones.T @ errors) / k
    
    return gradient
