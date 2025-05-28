import numpy as np


def create_boolean_vector(length_of_vector: int, num_of_ones: int) -> np.ndarray:
    """
    This function generates a boolean column vector of size N,
    containing num_of_ones random distributed 1, and the rest 0.
    :param length_of_vector: Length of the vector
    :param num_of_ones: Number of random ones
    :return: Boolean vector
    """
    if length_of_vector <= 0:
        raise ValueError("The length of the vector must be greater than 0.")
    if num_of_ones > length_of_vector:
        raise ValueError("The number of ones in the vector must be less than or equal to N.")
    x = np.zeros((length_of_vector, 1))
    selected_indices = np.random.choice(length_of_vector, num_of_ones, replace=False)
    x[selected_indices, :] = 1
    x = np.array(x, dtype=np.int32) # Convert to integer
    return x


def create_block_diag_matrix(x: np.ndarray, repeat: int = None) -> np.ndarray:
    """
    This function creates a block diagonal matrix from an input x.
    1. If x is a column vector, it first repeats the vector to form a matrix with a shape of (row, repeat),
    then it repeats the matrix to form a block diagonal matrix.
    2. If x is a matrix, it directly forms the block diagonal matrix.
    :param x: Input 2D numpy ndarray
    :param repeat: Number of times to repeat the vector, and None for matrix
    :return: Block diagonal matrix
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    if x.ndim != 2:
        raise TypeError("Input must be a 2D array.")

    row, column = x.shape
    if column == 1:  # X is a column vector
        if repeat is None:
            raise ValueError("`repeat` must be provided when the input is a column vector.")
        x = np.repeat(x, repeat, axis=1)  # Repeat the column vector to form a matrix with a shape of (row, repeat)
        column = repeat

    block_diag_matrix = np.zeros((column, row * column), dtype=np.complex128)

    for col in range(column):
        for r in range(row):
            block_diag_matrix[col, r + col * row] = x[r, col]

    return block_diag_matrix


def calculate_crb(sigmaR2, rc, L, Rx, a, a_diff):
    """
    Calculate Cramer-Rao Bound (CRB)
    :param sigmaR2: Sensing noise power
    :param rc: Radar sensing channel
    :param L: Number of snapshots
    :param Rx: Beamforming corvariance matrix
    :param a: Steering vector
    :param a_diff: Steering vector differential
    :return: root-CRB
    """
    M = a.shape[1] 
    SNR_r = np.abs(rc)**2 * L / sigmaR2  

    crb_values = np.zeros(M)
    for m in range(M):
        norm_a_diff = np.real(a_diff[:, m:m+1].T.conj() @ a_diff[:, m:m+1])
        norm_b_diag_W = np.real(a[:, m:m+1].T.conj() @ Rx @ a[:, m:m+1])
        crb_values[m] = 1 / (2 * SNR_r * norm_a_diff * norm_b_diag_W)

    return np.rad2deg(np.sqrt(np.mean(crb_values**2)))

