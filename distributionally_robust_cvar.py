import numpy as np
from scipy.stats import norm, invgamma
from gmm_creation import create_gmm
import matplotlib.pyplot as plt


class DistributionallyRobustCVaR:
    def __init__(self, gmm):
        self.gmm = gmm
        # gmm.means_: (E,1), gmm.covariances_: (E,1,1)
        self.mu    = gmm.means_.reshape(-1)                       # (E,)
        self.sigma = np.sqrt(gmm.covariances_.reshape(-1))        # (E,)

    def calculate_var(self, mu, sigma, alpha=0.95):
        """
        Calculate Value at Risk (VaR) for a normal distribution.
        """
        var = mu + sigma * norm.ppf(alpha)
        return var

    def calculate_cvar(self, mu, sigma, alpha=0.95):
        """
        Calculate Conditional Value at Risk (CVaR) for a normal distribution.
        """
        cvar = mu + sigma * (norm.pdf(norm.ppf(alpha)) / (1 - alpha))
        return cvar

    @staticmethod
    def _cvar_vec(mu, sigma, alpha):
        z    = norm.ppf(alpha)
        k    = norm.pdf(z) / (1.0 - alpha)
        return mu + sigma * k                                     # (E,)

    def compute_dr_cvar(self, alpha=0.95):
        """
        Compute the infimum of CVaR values from the GMM components.
        """        
        cvar_vec = self._cvar_vec(self.mu, self.sigma, alpha)
        idx_max  = int(cvar_vec.argmax())
        return float(cvar_vec[idx_max]), cvar_vec.tolist(), idx_max

    def is_within_boundary(self, boundary, alpha=0.95):
        """
        Check if the Distributionally Robust CVaR is within the specified boundary.
        """
        cvar_max = self._cvar_vec(self.mu, self.sigma, alpha).max()
        return cvar_max <= boundary

    @staticmethod
    def batch_within_boundary(mu_mat, sig2_mat, boundary, alpha=0.99):
        """
        Vectorised DR‑CVaR test for many (N,E) samples.
        mu_mat, sig2_mat : np.ndarray (N,E)
        returns           : np.ndarray bool mask (N,)
        """
        z    = norm.ppf(alpha)
        k    = norm.pdf(z) / (1 - alpha)
        worst_cvar = mu_mat + np.sqrt(sig2_mat) * k   # (N,E)
        worst_cvar = worst_cvar.max(axis=1)           # (N,)
        return worst_cvar <= boundary                 # (N,) Bool
    

def plot_gmm_with_cvar(gmm, cvar_values, dr_cvar_index):
    """
    Plot the GMM with individual components, CVaR boundaries, and DR_CVaR line.
    """
    x = np.linspace(gmm.means_.min() - 3, gmm.means_.max() + 3, 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x)
    responsibilities = gmm.predict_proba(x)
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    pdf = pdf * (1 / 3)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, '-k', label='GMM', linewidth=0.5)
    component_colors = ['lightcoral', 'lightgreen', 'lightblue']

    for i in range(pdf_individual.shape[1]):
        line, = plt.plot(x, pdf_individual[:, i], '-', label=f'GMM Component {i+1}', color=component_colors[i], linewidth=0.5)
        fill = plt.fill_between(x.flatten(), pdf_individual[:, i], color=component_colors[i], alpha=0.3)

    for i, cvar in enumerate(cvar_values):
        linestyle = '-'
        linewidth = 1 if i != dr_cvar_index else 2.5
        color = component_colors[i]
        plt.axvline(cvar, color=color, linestyle=linestyle, linewidth=linewidth, label=f'Component {i+1} CVaR')
        
        if i == dr_cvar_index:
            plt.annotate('DR_CVaR', xy=(cvar, 0.0), xytext=(cvar - 0.15, 0.03),
                         arrowprops=dict(facecolor=color, shrink=0.05),
                         horizontalalignment='right')

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Gaussian Mixture Model with Individual Components and CVaR Boundaries')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    # Define Gaussian distributions for means
    gaussians = [norm(loc=5, scale=1.5)]
    
    # Define Inverse-Gamma distributions for variances
    inv_gammas = [invgamma(a=2.5, scale=1.2)]
    
    # Create GMM
    gmm = create_gmm(gaussians, inv_gammas, num_samples=3)
    
    # Print the means and covariances of the created GMM
    print("GMM Means:", gmm.means_)
    print("GMM Covariances:", gmm.covariances_)

    # Initialize the Distributionally Robust CVaR filter
    cvar_filter = DistributionallyRobustCVaR(gmm)

    # Define a boundary for the CVaR
    boundary = 10

    # Compute the Distributionally Robust CVaR
    dr_cvar, cvar_values, dr_cvar_index = cvar_filter.compute_dr_cvar(alpha=0.95)
    print(f"Distributionally Robust CVaR: {dr_cvar}")

    # Check if the Distributionally Robust CVaR is within the specified boundary
    within_boundary = cvar_filter.is_within_boundary(boundary, alpha=0.95)
    print(f"Within Boundary: {within_boundary}")

    # Plot the GMM with individual components and CVaR boundaries
    plot_gmm_with_cvar(gmm, cvar_values, dr_cvar_index)