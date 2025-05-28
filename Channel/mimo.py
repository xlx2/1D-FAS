import numpy as np


class MIMO_Channel:
    def __init__(self, N: int, N_user: int):
        self.N = N
        self.N_user = N_user

    def get_channel(self):
        H = (np.random.randn(self.N, self.N_user) + 1j * np.random.randn(self.N, self.N_user)) / np.sqrt(2)

        return H
