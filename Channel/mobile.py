import numpy as np
from utils.math import dB2pow
from scipy.special import j0


class Mobile_Channel:
    def __init__(self, Ny: int, N_user: int, Nx: int=1, sigma2=dB2pow(0), pos_x=1, pos_y=1):
        self.Nx = Nx
        self.Ny = Ny
        self.N = Nx * Ny
        self.N_user = N_user
        self.sigma2 = sigma2
        self.pos_x = pos_x
        self.pos_y = pos_y
        
        # Create 2D indices
        nty, ntx = np.meshgrid(range(1, self.Ny + 1), range(1, self.Nx + 1))
        self.ntx = ntx.ravel()
        self.nty = nty.ravel()

        # Compute spatial correlation matrix
        self.J = self._compute_spatial_correlation()

        # Eigenvalue decomposition
        Ltx, self.Utx = np.linalg.eig(self.J)
        Ltx[Ltx < 0] = 0
        self.Ltx = np.diag(Ltx)

    def _compute_spatial_correlation(self):
        d1 = np.zeros((self.N, self.N))
        d2 = np.zeros((self.N, self.N))

        for i in range(self.N):
            if self.Nx != 1:
                d1[:, i] = np.abs(self.ntx[i] - self.ntx) / (self.Nx - 1) * self.Wx
            d2[:, i] = np.abs(self.nty[i] - self.nty) / (self.Ny - 1) * self.Wy

        d = np.sqrt(d1 ** 2 + d2 ** 2)  # Total distance
        J = self.sigma2 * j0(2 * np.pi * d)  # Spatial correlation

        return J

    def get_channel(self):
        g = (np.random.randn(self.N, self.N_user) + 1j * np.random.randn(self.N, self.N_user)) / np.sqrt(2)
        h = np.zeros((self.N, self.N_user), dtype=complex)

        for k in range(self.N_user):
            h[:, k] = np.conj(g[:, k].T) @ np.sqrt(self.Ltx.T) @ np.conj(self.Utx.T)

        return h
