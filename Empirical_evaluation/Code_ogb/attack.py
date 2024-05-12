"""
Main attack that are used in the evaluation
"""

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import torch.distributions as tdist
from torch.distributions.normal import Normal

class RandomNoise():
    """
    Random Noise attack
    ---
    Budget : Budget of the attack to be generated
    """
    def __init__(self, noise_ratio):
        self.noise_ratio = noise_ratio

    def perturb(self, data):
        x, num_nodes, num_feat = data.x, data.num_nodes, data.num_features

        loc = torch.zeros(num_feat).to(data.x.device)

        normal = Normal(loc, self.noise_ratio / np.sqrt(num_feat))
        noise = normal.sample((num_nodes, ))

        return  noise + data.x

class PGD():
    """
    Proximal Gradient attack
    adapted from : https://github.com/DSE-MSU/DeepRobust
    ---
    Budget : Budget of the attack to be generated
    epoch_iter : Number of PGD iterations to be used
    """

    def __init__(self, model_local, data_local, norm_adj, budget, epoch_iter = 50):
        self.model_local = model_local
        self.data_local = data_local
        self.budget = budget
        self.epoch_iter = epoch_iter
        self.budget = budget
        self.norm_adj = norm_adj

    def attack(self):
        self.model_local.eval()

        perturb = Parameter(torch.zeros(self.data_local.x.shape[0], \
                    self.data_local.x.shape[1])).to(self.data_local.x.device)

        for t in range(self.epoch_iter):

            temp_x = self.data_local.x + perturb

            out = self.model_local(temp_x, self.norm_adj)
            loss = F.cross_entropy(out[self.data_local.train_mask], self.data_local.y[self.data_local.train_mask])

            x_grad = torch.autograd.grad(loss, perturb)[0]

            lr = self.epoch_iter / np.sqrt(t+1)
            perturb.data.add_(lr * x_grad)

            perturb.data.copy_(torch.clamp(perturb.data, min=0, max=1))

        return self.project_perturb(perturb)

    def project_perturb(self, perturbation):
        norm_val = (self.budget * self.data_local.x.norm()) /  perturbation.norm()
        perturbation.data = (perturbation.data * norm_val)

        return perturbation



if __name__ == "__main__":
    pass
