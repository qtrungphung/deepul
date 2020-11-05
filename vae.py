from numpy.core.fromnumeric import repeat
from numpy.core.numeric import outer
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE2D(torch.nn.Module):

    def __init__(self):
        super(VAE, self).__init__
        self.decoder = nn.Sequential(nn.Linear(2, 2, bias=True),
                                     nn.Linear(2,2,True))
        self.encoder = nn.Sequential(nn.Linear(1, 2, bias=True),
                                     nn.Linear(2,2,True),
                                     nn.Linear(2,2,True))
        self.fc_mu = nn.Linear(2,1,bias=True)
        self.fc_logvar = nn.Linear(2, 1, True)
        self.mu = 0
        self.logvar = 0

    def encode(self, x):
        """latent z is gaussian, with mean mu and log sigma logvar
        mu = fc_mu(encode(x))
        logvar = fc_logvar(encode(x))
        """
        code = self.encode(x)
        self.mu = self.fc_mu(code)
        self.logvar = self.fc_logvar(code)
        return latent_mu, latent_logvar

    def reparameterization(self, mu, logvar):
        """generate a latent z from Gaussian
        mean mu and logvar"""

        std = torch.exp(logvar)
        return mu + std*torch.randn(size=len(mu))

    def decode(self, latent):
        """decode a latent z to x"""
        out = self.decoder(latent)
        return out

    def forward(self, x):
        """VAE forward: encode, reparam, decode"""
        mu, logvar = self.encode(x)
        z = self.reparameterization(mu, logvar)
        out = self.encode(z)
        return out

    def generate(self, num):
        """
        sample latent distribution and
        generate num number of data points
        """
        


        


