import numpy as np
from typing import Optional

def pow2dB(x):
    return 10 * np.log10(x)


def dB2pow(x):
    return 10 ** (x / 10)


def pow2dBm(x):
    return 10 * np.log10(x / 1e-3)


def dBm2pow(x):
    return 10 ** (x / 10) / 1e3


def eigenvalue_decomposition(xxH: np.ndarray, compress: Optional[int]=None, expand: Optional[int]=None) -> np.ndarray:
    """
    This function decomposes the semi-definite hermitian matrix xxH into a matrix x,
    where the eigenvectors and square roots of the eigenvalues. x @ xH = xxH
    :param xxH: Semi-definite hermitian matrix
    :param compress: Keep only the first K eigenvectors and eigenvalues. If None, all eigenvectors and eigenvalues are kept.
    :param expand: Number of zero columns to append to the last column of x. If None, no columns are appended.
    :return: x
    """
    if not isinstance(xxH, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    if xxH.ndim != 2 or xxH.shape[0] != xxH.shape[1]:
        raise ValueError("Input must be a square matrix.")
    if not np.allclose(xxH, xxH.T.conj()):
        raise ValueError("Input matrix must be Hermitian (symmetric if real-valued).")
    
    if compress is not None:
        if compress <= 0 or compress > xxH.shape[0]:
            raise ValueError("K must be between 1 and matrix dimension.")
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(xxH)

    # 将负特征值设为零(处理数值误差)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # 按特征值大小降序排序
    sorted_indices = np.argsort(eigenvalues)[::-1]  # 降序排列
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # 只保留前K个最大的特征值和对应的特征向量
    topK_eigenvalues = sorted_eigenvalues[:compress]
    topK_eigenvectors = sorted_eigenvectors[:, :compress]
    
    # 构建分解矩阵
    x = topK_eigenvectors @ np.diag(np.sqrt(topK_eigenvalues))
    
    # 如果需要，追加零列
    if expand is not None and expand > 0:
        zeros_to_append = np.zeros((x.shape[0], expand), dtype=x.dtype)
        x = np.hstack([x, zeros_to_append])
    
    return x