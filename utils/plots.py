import numpy as np
import matplotlib.pyplot as plt
from utils.math import pow2dB
from utils.antenna_array import *
from typing import Optional
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 5,
})


def ula_beampattern(N:int , type:str, Rx: Optional[np.ndarray]=None, spacing: float = 0.5,
                    isNormalized:bool=False, save_dir_path: Optional[str]=None):
    """
    Draw the beampatterning plot.
    :param N: Number of antennas
    :param type: Type of the antenna array
    :param Rx: Beamforming matrix
    :param spacing: Spacing between antennas
    :param isNormalized: Whether to normalize the beamforming matrix.
    :param save_path: Path to save the plot.
    :return: None
    """
    if Rx is None:
        Rx = np.ones((N, 1)) @ np.ones((1, N))
    if not isinstance(Rx, np.ndarray) or Rx.ndim != 2 or Rx.shape[0] != N or Rx.shape[1] != N:
        raise TypeError("Input must be a 2D numpy ndarray of size (N, N)")
    angles = np.arange(-90, 90.1, 0.1)
    if type == 'mimo':
        a_theta = get_mimo_ula_steering_vector(theta=angles, N=N, spacing=spacing)
    elif  type == 'fas':
        a_theta = get_fas_ula_steering_vector(theta=angles, N=N, spacing=spacing)
    power_matrix = np.real(np.diag(a_theta.T.conj() @ Rx @ a_theta) / np.trace(Rx))
    if isNormalized:
        power_matrix = power_matrix / np.max(power_matrix)
    power_dB_matrix = pow2dB(power_matrix)
    power_dB_matrix = np.clip(power_dB_matrix, -50, np.inf)

    plt.figure()
    plt.plot(angles, power_dB_matrix, '-', color=(0.8706, 0.3451, 0.1686), label='Uniform linear array')
    plt.xlabel(r"Angle $(\theta^{\circ})$")
    plt.ylabel(r"Power $(\text{dB})$")
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()
    plt.grid(True)
    if save_dir_path is not None:
        plt.savefig(save_dir_path + f'/ula_beampattern.png', format="png", dpi=300)
        plt.savefig(save_dir_path + f'/ula_beampattern.eps', format="eps")
    plt.show()

def upa_beampattern(Ny:int, Nz:int, Rx: Optional[np.ndarray]=None, spacing:float=0.5, 
                    isNormalized:bool=False, save_dir_path: Optional[str]=None):
    """
    Draw the beampatterning plot.
    :param Ny: Number of transmitters in the y-direction.
    :param Nz: Number of transmitters in the z-direction.
    :param Rx: Beamforming matrix
    :param spacing: Spacing between antennas
    :param isNormalized: Whether to normalize the beamforming matrix.
    :param save_path: Path to save the plot.
    :return: None
    """
    N = Ny * Nz
    if Rx is None:
        Rx = np.ones((N, 1)) @ np.ones((1, N))
    if not isinstance(Rx, np.ndarray) or Rx.ndim != 2 or Rx.shape[0] != N or Rx.shape[1] != N:
        raise TypeError("Input must be a 2D numpy ndarray of size (Nx * Ny, Nx * Ny)")
    theta_range = np.arange(0, 91, 1)  # Elevation (0° to 90°)
    phi_range = np.arange(-90, 91, 1)  # Azimuth (-180° to 180°)
    theta_grid, phi_grid = np.meshgrid(theta_range, phi_range)
    power_matrix = np.zeros_like(theta_grid, dtype=np.float32)
    for i, theta in enumerate(theta_range):
        for j, phi in enumerate(phi_range):
            a = get_mimo_upa_steering_vector(phi, theta, Ny, Nz, spacing)
            power_matrix[j, i] = np.real(a.T.conj() @ Rx @ a / np.trace(Rx))
    if isNormalized:
        power_matrix = power_matrix / np.max(power_matrix)
    power_dB_matrix = pow2dB(power_matrix)
    power_dB_matrix = np.clip(power_dB_matrix, -50, np.inf)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(phi_grid, theta_grid, power_dB_matrix, cmap='viridis', alpha=0.85)
    ax.set_xlabel(r'Azimuth $(\phi^{\circ})$')
    ax.set_ylabel(r'Elevation $(\theta^{\circ})$')
    ax.set_zlabel(r'Power $(\text{dB})$')
    plt.grid(True)
    plt.tight_layout()
    ax.view_init(elev=30, azim=45)
    if save_dir_path is not None:
        plt.savefig(save_dir_path + f'/upa_beampattern.png', format="png", dpi=300)
        plt.savefig(save_dir_path + f'/upa_beampattern.eps', format="eps")
    plt.show()