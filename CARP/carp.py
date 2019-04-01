import cvxpy as cp 
import numpy as np 
import pandas as pd 

class MARPR:
    """
    Python implementation of the CARP model, proposed as part of the paper produced by 
    Team 9 for the DataOpen Championship
    """
    def __init__(self, include_lambda = True):
        """
        Args:
            - include_lamba: Bool, weither or not to include the auto-regressive term
        """
        self.XNl = False
        self.verbose = False
        self.include_lambda = include_lambda
        
    def _set_XNl(self, X_theta, X_lambda, N):
        """Check the input tensors and make some shorthands notations"""
        # check dims
        assert X_theta.shape[:2] == N.shape
        assert X_lambda.shape[:2] == N.shape
        assert X_theta.shape[0] == X_lambda.shape[0]
        assert X_theta.shape[1] == X_lambda.shape[1]
        
        self.K_theta = X_theta.shape[2] # number of regressors
        self.K_lambda = X_lambda.shape[2]
        self.C = X_theta.shape[0] # number of countries
        self.T = X_theta.shape[1] - 1 # number of time periods
        
        # remove first time period for X as we will need N_t-1
        X_theta = X_theta[:,1:,:]
        X_lambda = X_lambda[:,1:,:]
        # We won't need the last value of N as a regressor too
        Nl = N[:,:-1]
        # Remove first obs to match the regressors
        self.N = N[:,1:]
        self.Nl = N[:,:-1]
        self.X_theta = X_theta
        self.X_lambda = X_lambda
        
    def _neg_log_likelihood(self, theta, lam):
        """ Negative log-likelihood to be used by CVXPY"""
        pX_aslist = [x_theta * theta.T + cp.multiply(x_lambda * lam.T, Nt.reshape(-1,1)) \
                     for (x_theta, x_lambda, Nt) in zip(self.X_theta, self.X_lambda, self.Nl)]
        
        
        loglik_aslist = [p_x - cp.multiply(Nt.reshape(-1,1), cp.log(p_x)) \
                         for p_x, Nt in zip(pX_aslist, self.N)]
        
        loglik = cp.vstack(loglik_aslist)
        
        return cp.sum(loglik)
    
    def _optimize(self):
        """
        CVXPY minimization of the negative log-likelihood
        """
        theta = cp.Variable((1, self.K_theta))
        lam = cp.Variable((1, self.K_lambda))
        
        loss = self._neg_log_likelihood(theta, lam)
        
        if self.include_lambda:
            prob = cp.Problem(cp.Minimize(loss))
        else:
            prob = cp.Problem(cp.Minimize(loss), [lam==0])
            
        prob.solve(verbose=self.verbose, solver='SCS')
        self.theta_, self.lambda_ = theta.value.reshape(-1), lam.value.reshape(-1)
        self.l_ = - prob.value
        
    def _populate_Fisher_information_matrix(self):
        """Compute the Fisher information matrix, to derive t-tests and confidence intervals"""
        # shorthands:
        X_theta, X_lambda, N, theta, lam, Nl = self.X_theta, self.X_lambda, self.N, self.theta_, self.lambda_, self.Nl
        # We loop because it's not expensive but we want to be sure of the result
        J = np.zeros((self.K_theta + self.K_lambda, self.K_theta + self.K_lambda))
        for i in range(self.C):
            for t in range(self.T):
                
                # for theta
                for p in range(self.K_theta):
                    for q in range(self.K_theta):
                        J[p,q] += (N[i, t] * X_theta[i,t,p] * X_theta[i,t,q]) / \
                        (np.dot(X_theta[i,t], theta.T) + np.dot(X_lambda[i,t], lam.T) * Nl[i,t])**2
                        
                # for lambda / theta
                for p in range(self.K_theta):
                    for q in range(self.K_lambda):
                        v = (N[i, t]*Nl[i,t] * X_theta[i,t,p] *  X_lambda[i,t,q]) / \
                        (np.dot(X_theta[i,t], theta.T) + np.dot(X_lambda[i,t], lam.T) * Nl[i,t])**2
                        J[p,self.K_theta+q] += v
                        J[self.K_theta+q,p] += v
                        
                # for lambda
                for p in range(self.K_lambda):
                    for q in range(self.K_lambda):
                        J[self.K_theta+p,self.K_theta+q] += \
                        (N[i, t]*Nl[i, t]**2 * X_lambda[i,t,p] *  X_lambda[i,t,q]) / \
                        (np.dot(X_theta[i,t], theta.T) + np.dot(X_lambda[i,t], lam.T) * Nl[i,t])**2

        self.J = J / (self.T * self.C)
        
    def _get_confidence_intervals(self):
        c = 1.96
        I = np.linalg.inv(self.J) / (self.C * self.T)
        
        self.theta_confidence = np.zeros(self.K_theta)
        self.lambda_confidence = np.zeros(self.K_lambda)
        self.theta_tstat = np.zeros(self.K_theta)
        self.lambda_tstat = np.zeros(self.K_lambda)
        
        for p in range(self.K_theta):
            self.theta_confidence[p] = c * np.sqrt(I[p,p])
            self.theta_tstat[p] = self.theta_[p] / np.sqrt(I[p,p])
        for p in range(self.K_lambda):
            self.lambda_confidence[p] = c * np.sqrt(I[self.K_theta + p, self.K_theta + p])
            self.lambda_tstat[p] = self.lambda_[p] / np.sqrt(I[self.K_theta + p, self.K_theta + p])
        
        self.I = I
        
    def fit(self, X_theta, X_lambda, N):
        """Function to call the calibrate the model
        Args:
            - X_theta: Regressors to use for the constant component.
                A i * t * k tensor, where i is the number of panel (countries in our paper),
                t is the number of temporal observations, and k the number of features.
            - X_lambda: Same as for X_theta, regressors to use for the auto-regressive component
            - N: A i * t target matrix (representing the number of infection in our paper)
        """
        self._set_XNl(X_theta, X_lambda, N)
        self._optimize()
        self._populate_Fisher_information_matrix()
        self._get_confidence_intervals()
        
    def to_df(self, theta_features_names, lambda_features_names):
        """Export the results as a readable pandas dataframe
        Args:
            - theta_features_names: list of strings, the names of the theta regressors
            - lambda_features_names: list of strings, the names of the lambda regressors
        """
        param = ['Theta' for _ in theta_features_names] + \
                ['Lambda' for _ in lambda_features_names]
        features_names = theta_features_names + lambda_features_names
        _95_conf = list(self.theta_confidence) + list(self.lambda_confidence)
        tstats = list(self.theta_tstat) + list(self.lambda_tstat)
        return pd.DataFrame.from_dict({
            'Param':param,
            'Feature' : np.array(features_names),
            'Value': list(self.theta_) + list(self.lambda_),
            'tstat': np.array(tstats),
            '95% confidence':np.array(_95_conf)
        })
    
    def freedom(self):
        """Return the number of degrees of freedom of the model"""
        if not self.include_lambda:
            return self.K_theta
        return self.K_theta + self.K_lambda
    