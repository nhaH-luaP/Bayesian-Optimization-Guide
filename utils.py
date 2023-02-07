"""
    Utils folder that contains important functions/modules used in main.ipynb
    Brief summary of contents:
"""

import torch
import gpytorch
import matplotlib.pyplot as plt
import torch.nn.functional as F

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class PlotBO():
    def __init__(self):
        self.init = True

    def plot_bo_1d(self, param_space, domain, targets, x_train, y_train, model, acqf, iteration, beta):
        # Plot ground truth function as a black line
        fig, axis = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 4), constrained_layout=True)
        fig.suptitle('Epoch '+str(iteration) + ' (Beta='+str(beta)+')', fontsize=12)

        # Plotting the objective function
        axis[0].plot(param_space, targets, c='black')
        axis[0].scatter(x_train, y_train, c='blue')
        axis[0].set_ylabel("objective")

        # Plot the Surrogate of the Objective function with confidence interval
        dist = model(param_space)
        mean, std = dist.mean.detach(), dist.stddev.detach()

        axis[1].scatter(x_train, y_train, label="Observations")
        axis[1].plot(param_space, mean, label="Mean prediction", c='black')
        axis[1].fill_between(
            param_space.ravel(),
            mean - 1.96 * std,
            mean + 1.96 * std,
            alpha=0.5,
            label=r"95% confidence interval",
        )
        axis[1].set_ylabel("surrogate")
        axis[1].scatter(x_train, y_train, c='blue')

        # Plot the Acquisition Function
        acqf_space = acqf(param_space.reshape(-1, 1, 1)).detach()
        axis[2].plot(param_space.reshape(-1, 1), acqf_space, c='black')
        axis[2].set_ylabel("acq-func")
        axis[2].set_xlabel("x")

        # Adding a vline for the last chosen sample
        y_dom = torch.max(torch.abs(torch.stack((1.96*std.flatten(), acqf_space.flatten(), targets.flatten())))) + 0.1
        for ax in axis:
            ax.vlines(x_train[-1], -y_dom, y_dom)
            ax.set_xlim((-domain, domain))
            ax.set_ylim((-y_dom, y_dom))

        plt.show()


    def plot_bo_2d(self, domain_2d, xx, yy, param_space_2d, targets_2d, acqf, model, x_train, iteration, beta):
        fig, axis = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(7,5), dpi=80, facecolor='silver', constrained_layout=True)
        fig.suptitle('Epoch '+str(iteration) + ' (Beta='+str(beta)+')', fontsize=12)

        # Calculate Values of the param space for objective function, acquisition function and variance of gaussian process thats being used as a surrogate
        target_space = F.normalize(targets_2d.view(xx.shape))
        ucb_space = F.normalize(acqf(param_space_2d.reshape(-1, 1, 2)).view(xx.shape).detach())
        model_output = model(param_space_2d)
        var_space = F.normalize(model_output.stddev.detach().view(xx.shape)**2)
        mean_space = F.normalize(model_output.mean.detach().view(xx.shape))

        # Here the objective function will be plottet with sampled points
        con = axis[0][0].contourf(xx, yy, target_space, alpha=.8, zorder=-1, cmap='plasma')
        axis[0][0].set_title("TARGET")
        plt.colorbar(con, ax=axis[0][0])

        # Here the Acquisition function values will be plottet
        con = axis[0][1].contourf(xx, yy, ucb_space, alpha=.8, zorder=-1, cmap='plasma')
        axis[0][1].set_title("ACQF")
        plt.colorbar(con, ax=axis[0][1])

        # Variance of the Gaussian Process
        con = axis[1][0].contourf(xx, yy, var_space, alpha=.8, zorder=-1, cmap='plasma')
        axis[1][0].set_title("GP-VAR")
        plt.colorbar(con, ax=axis[1][0])

        # Mean of the Gaussian Process
        con = axis[1][1].contourf(xx, yy, mean_space, alpha=.8, zorder=-1, cmap='plasma')
        axis[1][1].set_title("GP-MEAN")
        plt.colorbar(con, ax=axis[1][1])

        for i in range(2):
            for j in range(2):
                axis[i][j].vlines(x_train[-1][0], ymin=-domain_2d, ymax=domain_2d)
                axis[i][j].hlines(x_train[-1][1], xmin=-domain_2d, xmax=domain_2d)
                axis[i][j].scatter(x_train[:,0], x_train[:,1], c='black', s=5)

        # Additional cross for reference of which point has been sampled
        plt.show()