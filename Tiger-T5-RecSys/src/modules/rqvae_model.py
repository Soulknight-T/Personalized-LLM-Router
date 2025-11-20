import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class ResidualQuantizer(nn.Module):
    def __init__(self, num_quantizers, codebook_size, codebook_dim):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, codebook_dim) for _ in range(num_quantizers)]
        )

    @torch.no_grad()
    def initialize_codebooks(self, x):
        # K-means initialization for each residual stage
        residuals = x
        for i in range(self.num_quantizers):
            data = residuals.cpu().detach().numpy()
            n_samples = data.shape[0]

            if n_samples < self.codebook_size:
                n_clusters = n_samples
                kmeans = KMeans(n_clusters=n_clusters, n_init='auto', max_iter=100, random_state=0)
                kmeans.fit(data)
                centroids = torch.from_numpy(kmeans.cluster_centers_).to(residuals.device, dtype=residuals.dtype)
                self.codebooks[i].weight.data[:n_clusters].copy_(centroids)
                remaining = self.codebook_size - n_clusters
                std = data.std() * 0.1
                rand_fill = torch.randn(remaining, self.codebook_dim, device=residuals.device, dtype=residuals.dtype) * std
                self.codebooks[i].weight.data[n_clusters:].copy_(rand_fill)
                indices = torch.from_numpy(kmeans.labels_).to(residuals.device)
            else:
                kmeans = KMeans(n_clusters=self.codebook_size, n_init='auto', max_iter=100, random_state=0)
                kmeans.fit(data)
                centroids = torch.from_numpy(kmeans.cluster_centers_).to(residuals.device, dtype=residuals.dtype)
                self.codebooks[i].weight.data.copy_(centroids)
                indices = torch.from_numpy(kmeans.labels_).to(residuals.device)

            # compute next residual
            residuals = residuals - self.codebooks[i](indices)

    def forward(self, x):
        residuals = x
        quantized_list = []
        total_commit, total_vq = 0.0, 0.0

        for codebook in self.codebooks:
            # distance to each codeword
            distances = (
                torch.sum(residuals ** 2, dim=1, keepdim=True)
                - 2 * residuals @ codebook.weight.T
                + torch.sum(codebook.weight ** 2, dim=1).unsqueeze(0)
            )
            indices = torch.argmin(distances, dim=1)
            q = codebook(indices)

            # VQ + commitment losses
            total_commit += F.mse_loss(residuals, q.detach())
            total_vq += F.mse_loss(residuals.detach(), q)

            # straight-through estimator
            q_ste = residuals + (q - residuals).detach()
            quantized_list.append(q_ste)

            # update residual
            residuals = residuals - q

        quantized_sum = torch.stack(quantized_list, dim=0).sum(dim=0)
        loss = total_commit + 0.25 * total_vq
        return quantized_sum, loss

    @torch.no_grad()
    def get_codes(self, x):
        # encode to discrete codes
        residuals = x
        codes = []
        for codebook in self.codebooks:
            distances = (
                torch.sum(residuals ** 2, dim=1, keepdim=True)
                - 2 * residuals @ codebook.weight.T
                + torch.sum(codebook.weight ** 2, dim=1).unsqueeze(0)
            )
            indices = torch.argmin(distances, dim=1)
            codes.append(indices)
            residuals = residuals - codebook(indices)
        return torch.stack(codes, dim=1)

    @torch.no_grad()
    def get_vectors_from_codes(self, codes):
        # decode from discrete codes
        xq = 0.0
        for i, codebook in enumerate(self.codebooks):
            xq += codebook(codes[:, i])
        return xq


class RQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, codebook_dim, num_quantizers, codebook_size):
        super().__init__()
        # simple MLP encoder-decoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.pre_quant = nn.Linear(hidden_dim, codebook_dim)

        self.quantizer = ResidualQuantizer(num_quantizers, codebook_size, codebook_dim)

        self.post_quant = nn.Linear(codebook_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, input_dim)
        )

    def encode(self, x):
        return self.pre_quant(self.encoder(x))

    def quantize(self, x_latent):
        return self.quantizer(x_latent)

    def decode(self, xq):
        return self.decoder(self.post_quant(xq))

    def forward(self, x):
        # encode → quantize → decode
        latent = self.encode(x)
        q, commit_loss = self.quantize(latent)
        recon = self.decode(q)
        recon_loss = F.mse_loss(recon, x)
        return recon, recon_loss, commit_loss
