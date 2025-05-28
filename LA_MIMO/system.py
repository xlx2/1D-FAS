from utils.plots import ula_beampattern
import cvxpy as cp
from utils.utils import calculate_crb


class MIMO_1D_System:
    def __init__(self, **parameters):
        for key, value in parameters.items():
            setattr(self, key, value)

    def run(self, verbose:bool=False, isPlot:bool=False):
        """
        Run the MIMO system model
        Parameters
        verbose : bool, optional
            Whether to print the results, by default False
        Returns
        Rx  : np.ndarray
            Beamforming Corviance Matrix
        crb : float
            The CRB of the MIMO system
        """

        # Define CVXPY Variables
        Rx = cp.Variable((self.N, self.N), hermitian=True)
        W = [cp.Variable((self.N, self.N), hermitian=True) for _ in range(self.K)]
        t = cp.Variable()

        # Constraints
        constraints = []

        # Objective Function Constraints for Sensing
        for m in range(self.M):
            constraints.append(cp.real(self.a[:, m:m+1].T.conj() @ Rx @ self.a[:, m:m+1]) >= t)

        # Power Constraint
        constraints.append(cp.real(cp.trace(Rx)) <= self.P)

        # Qos Constraints
        for k in range(self.K):
            lhs = (1 + 1 / self.Gamma) * cp.real(self.H[:, k:k+1].T.conj() @ W[k] @ self.H[:, k:k+1])
            rhs = cp.real(self.H[:, k:k+1].T.conj() @ Rx @ self.H[:, k:k+1]) + self.sigmaC2
            constraints.append(rhs - lhs <= 0)

        # SDR Constraints
        constraints.append((Rx - sum(W)) >> 0)
        constraints.append(Rx >> 0)
        for k in range(self.K):
            constraints.append(W[k] >> 0)

        # Define the objective function
        objective = cp.Maximize(t)

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=verbose)
        assert problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE], "Optimization failed.\n"
        if isPlot:
            # Plot the beampattern
            ula_beampattern(N=self.N, type='mimo', Rx=Rx.value, spacing=self.spacing)
        crb = calculate_crb(self.sigmaR2, self.rc, self.L, Rx.value, self.a, self.a_diff)
        return Rx.value, crb