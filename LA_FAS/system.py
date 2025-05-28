from matplotlib import pyplot as plt
from utils.plots import ula_beampattern
from tqdm import tqdm
import numpy as np
import cvxpy as cp
from utils.utils import *
from utils.math import *


class FA_1D_System:
    def __init__(self, **parameters):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.crb_list = []

    def init_problem(self, verbose=False):
        """
        Initialize lambda and W using SDR.

        :param verbose: Whether to print iteration logs
        :return: Initialized lambda(0), W(0)
        """
        lambda_old = np.ones((self.N, 1)) * (self.Nt / self.N)

        # Define CVXPY Variables
        Rx = cp.Variable((self.N, self.N), hermitian=True)
        W = [cp.Variable((self.N, self.N), hermitian=True) for _ in range(self.K)]
        t = cp.Variable()

        # Constraints
        constraints = []

        # Objective Function Constraints for Sensing
        for m in range(self.M):
            constraints.append(cp.real(self.a[:, m:m+1].T.conj() @ np.diag(lambda_old[:, 0]) @ Rx @ np.diag(lambda_old[:, 0]).T.conj() @ self.a[:, m:m+1]) >= t)

        # Power Constraint
        constraints.append(cp.real(cp.trace(np.diag(lambda_old[:, 0]) @ Rx @ np.diag(lambda_old[:, 0].T.conj()))) <= self.P)

        # Qos Constraints
        for k in range(self.K):
            lhs = (1 + 1 / self.Gamma) * cp.real(self.H[:, k:k+1].T.conj() @ np.diag(lambda_old[:, 0]) @ W[k] @ np.diag(lambda_old[:, 0]).T.conj() @ self.H[:, k:k+1])
            rhs = cp.real(self.H[:, k:k+1].T.conj() @ np.diag(lambda_old[:, 0]) @ Rx @ np.diag(lambda_old[:, 0]).T.conj() @ self.H[:, k:k+1]) + self.sigmaC2
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
        assert problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE], "Optimization failed at initialization.\n"
        
        W_old = eigenvalue_decomposition(Rx.value, expand=self.K)
        return lambda_old, W_old

    def update_W(self, lambda_old, W_old, verbose=False):
        """
        Update the W value.
        :param lambda_old: Obtained lambda value from the previous iteration.
        :param verbose: Whether to print iteration logs
        :return: Updated W value.
        """
        # Define CVXPY Variables
        W = cp.Variable((self.N, self.K+self.N), complex=True)
        t = cp.Variable()

        # Constraints
        constraints = []
        for m in range(self.M):
            constraints.append(cp.real(self.a[:, m:m+1].T.conj() @ np.diag(lambda_old[:, 0]) @ (W @ W_old.T.conj()) @ np.diag(lambda_old[:, 0]).T.conj() @ self.a[:, m:m+1]) >= t)

        # Power Constraint
        constraints.append(cp.norm(np.diag(lambda_old[:, 0]) @ W, 'fro') <= np.sqrt(self.P))

        # Qos Constraints
        for k in range(self.K):
            lhs = np.sqrt(1 + 1 / self.Gamma) * cp.real(self.H[:,k:k+1].T.conj() @ np.diag(lambda_old[:, 0]) @ W[:,k:k+1])
            rhs = cp.norm(np.vstack([create_block_diag_matrix((self.H[:, k:k + 1].T.conj() @ np.diag(lambda_old[:, 0])).T, repeat=self.K + self.N), np.zeros([1, self.N * (self.K + self.N)])]) @ W.reshape((W.size, 1), 'F') + np.vstack([np.zeros([self.K + self.N, 1]), np.array([[np.sqrt(self.sigmaC2)]])]), 2)
            constraints.append(rhs - lhs <= 0)
        
        # Define the objective function
        objective = cp.Maximize(t)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=verbose)

        assert problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE], "Optimization failed at W update.\n"
        return W.value

    def update_lambda(self, lambda_pre, W_old, rho1, rho2, verbose=False):
        """
        Update the lambda value.
        :param lambda_pre: Obtained lambda value from the previous iteration.
        :param W_old: Obtained W value from the previous iteration.
        :param verbose: Whether to print iteration logs
        :return: Updated lambda value
        """
        # Define CVXPY Variables
        lambda_old = cp.Variable((self.N, 1))
        t = cp.Variable()

        # Constraints
        constraints = []

        # Objective Function Constraints for Sensing
        for m in range(self.M):
            g = 2 * np.diag(self.a[:, m].conj()) @ (W_old @ W_old.T.conj()) @ np.diag(self.a[:, m]) @ lambda_pre + rho1 * (2 * lambda_pre - np.ones((self.N, 1)))
            constraints.append(cp.real(lambda_old.T @ g) - rho2 * cp.abs(np.ones((1, self.N)) @ lambda_old - self.Nt) >= t)

        constraints.append(cp.norm(cp.diag(lambda_old[:, 0]) @ W_old, 'fro') <= np.sqrt(self.P))

        # Qos Constraints
        for k in range(self.K):
            lhs = np.sqrt(1 + 1/self.Gamma) * cp.real(self.H[:,k:k+1].T.conj() @ cp.diag(lambda_old[:, 0]) @ W_old[:,k:k+1])
            rhs = cp.norm(lambda_old.T @ np.hstack([np.diag(self.H[:, k].conj()) @ W_old, np.zeros([self.N, 1])]) +
                        np.hstack([np.zeros([1, self.N + self.K]), np.array([[np.sqrt(self.sigmaC2)]])]), 2)
            constraints.append(rhs - lhs <= 0)
            
        # Selected Number Constraint
        constraints.append(0 <= lambda_old)
        constraints.append(lambda_old <= 1)

        # Define the objective function
        objective = cp.Maximize(t)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=verbose)
        
        # assert problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE], "Optimization failed at lambda update.\n"
        return lambda_old.value
    
    def refine_lambda(self, lambda_old, verbose:bool=False):
        """
        Refine the lambda value.
        :param lambda_old: Obtained lambda value from the previous iteration.
        :param verbose: Whether to print iteration logs
        :return: Refined lambda value
        """
        idx = np.argsort(lambda_old[:, 0])[-self.Nt:][::-1]
        lambda_old = np.zeros_like(lambda_old)
        lambda_old[idx, 0] = 1

        # Define CVXPY Variables
        Rx = cp.Variable((self.N, self.N), hermitian=True)
        W = [cp.Variable((self.N, self.N), hermitian=True) for _ in range(self.K)]
        t = cp.Variable()

        # Constraints
        constraints = []

        # Objective Function Constraints for Sensing
        for m in range(self.M):
            constraints.append(cp.real(self.a[:, m:m+1].T.conj() @ np.diag(lambda_old[:, 0]) @ Rx @ np.diag(lambda_old[:, 0]).T.conj() @ self.a[:, m:m+1]) >= t)

        # Power Constraint
        constraints.append(cp.real(cp.trace(np.diag(lambda_old[:, 0]) @ Rx @ np.diag(lambda_old[:, 0]).T.conj())) <= self.P)

        # Qos Constraints
        for k in range(self.K):
            lhs = (1 + 1 / self.Gamma) * cp.real(self.H[:, k:k+1].T.conj() @ np.diag(lambda_old[:, 0]) @ W[k] @ np.diag(lambda_old[:, 0]).T.conj() @ self.H[:, k:k+1])
            rhs = cp.real(self.H[:, k:k+1].T.conj() @ np.diag(lambda_old[:, 0]) @ Rx @ np.diag(lambda_old[:, 0]).T.conj() @ self.H[:, k:k+1]) + self.sigmaC2
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
        assert problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE], "Optimization failed at lambda refine process.\n"
            
        # W_old = eigenvalue_decomposition(Rx.value, self.K)
        return np.diag(lambda_old[:,0]) @ Rx.value @ np.diag(lambda_old[:,0])
    
    def run(self, MAX_ITER=100, verbose:bool=False, isPlot:bool=False):
        lambda_old, W_old = self.init_problem(verbose=verbose)
        rho1 = min([
            np.linalg.norm(self.a[:, 0:1].T.conj() @ np.diag(lambda_old[:, 0]) @ W_old, 2),
            np.linalg.norm(self.a[:, 1:2].T.conj() @ np.diag(lambda_old[:, 0]) @ W_old, 2),
            np.linalg.norm(self.a[:, 2:3].T.conj() @ np.diag(lambda_old[:, 0]) @ W_old, 2)
        ])
        rho2 = rho1**6
        with tqdm(total=MAX_ITER, desc="Optimizing", unit="iter", leave=False) as pbar:
            for iter in range(MAX_ITER):
                W_new = self.update_W(lambda_old, W_old, verbose=verbose)
                lambda_new = self.update_lambda(lambda_old, W_new, rho1=rho1, rho2=rho2, verbose=verbose)
                crb = calculate_crb(sigmaR2=self.sigmaR2, rc=self.rc, L=self.L, 
                                    Rx=np.diag(lambda_new[:,0]) @ W_new @ W_new.T.conj() @ np.diag(lambda_new[:,0]),
                                    a=self.a, a_diff=self.a_diff)
                self.crb_list.append(crb)
                
                W_old, lambda_old = W_new, lambda_new
                if iter > 0 and abs((self.crb_list[iter] - self.crb_list[iter-1]) / self.crb_list[iter]) < 1e-4:
                    print(f"\nConverged at iteration {iter}.")
                    break
                pbar.set_postfix({"crb": f"{self.crb_list[-1]:.4e}"})
                pbar.update(1)

        # Refine the lambda value
        Rx = self.refine_lambda(lambda_old, verbose=verbose)
        
        if isPlot:
            # plot records
            plt.figure()
            plt.plot(self.crb_list, label="CRB", linestyle="-.")
            plt.xlabel("Iteration")
            plt.legend()
            plt.grid(True)
            plt.show()
            # Plot the beampattern
            ula_beampattern(N=self.N, type='fas', Rx=Rx, spacing=self.spacing)

        return Rx, self.crb_list[-1]
            
    