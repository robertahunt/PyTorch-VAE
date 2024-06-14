import torch
import numpy as np
from ete3 import Tree
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class BatchMiner:
    def __init__(self, species, T):
        self.lower_cutoff = 0.5
        self.upper_cutoff = 1.4
        self.name = "distance"
        self.species = species
        self.species_dists = np.sqrt(64) * (1 - T)

    def __call__(
        self, batch, labels, tar_labels=None, return_distances=False, distances=None
    ):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(labels, list):
            labels = np.array(labels)
        bs, dim = batch.shape

        if distances is None:
            distances = self.pdist(batch.detach()).clamp(min=self.lower_cutoff)
        sel_d = distances.shape[-1]

        positives, negatives = [], []
        labels_visited = []
        anchors = []
        margins = []

        tar_labels = labels if tar_labels is None else tar_labels

        for i in range(bs):
            neg = tar_labels != labels[i]
            pos = tar_labels == labels[i]

            anchors.append(i)
            q_d_inv = self.inverse_sphere_distances(
                dim, bs, distances[i], tar_labels, labels[i]
            )
            neg_idx = np.random.choice(sel_d, p=q_d_inv)
            negatives.append(neg_idx)

            if np.sum(pos) > 0:
                # Sample positives randomly
                if np.sum(pos) > 1:
                    pos[i] = 0
                positives.append(np.random.choice(np.where(pos)[0]))
                # Sample negatives by distance
            neg_label = labels[neg_idx]
            anchor_label = labels[i]
            neg_species_idx = self.species.index(neg_label)
            anchor_species_idx = self.species.index(anchor_label)
            margin = self.species_dists[neg_species_idx, anchor_species_idx]
            margins.append(margin)

        # add pseudo distance
        # ie distance from genera to another species
        for i in range(bs):
            anchors.append(i)

            # q_d_inv = self.inverse_sphere_distances(
            #    dim, bs, distances[i], tar_labels, labels[i]
            # )

            # ensures all have some small probability so don't get stuck with all the same label
            # q_d_inv = q_d_inv + 0.01
            # q_d_inv = q_d_inv / q_d_inv.sum()

            q_d_inv = self.equal_distances(distances[i])

            idx_1, idx_2 = np.random.choice(sel_d, p=q_d_inv, size=2, replace=False)
            label_1, label_2 = labels[[idx_1, idx_2]]
            while label_1 == label_2:
                idx_1, idx_2 = np.random.choice(sel_d, p=q_d_inv, size=2, replace=False)
                label_1, label_2 = labels[[idx_1, idx_2]]
            dist_1, dist_2 = (
                self.species_dists[anchor_species_idx, idx_1],
                self.species_dists[anchor_species_idx, idx_2],
            )

            if dist_1 < dist_2:
                positives.append(idx_1)
                negatives.append(idx_2)
                margins.append(
                    dist_2 - dist_1 / 2
                )  # effective distance between group species01 and species 2
            else:
                positives.append(idx_2)
                negatives.append(idx_1)
                margins.append(
                    dist_1 - dist_2 / 2
                )  # effective distance between group species01 and species 2

        sampled_triplets = [
            [a, p, n, m] for a, p, n, m in zip(anchors, positives, negatives, margins)
        ]

        if return_distances:
            return sampled_triplets, distances
        else:
            return sampled_triplets

    def equal_distances(self, anchor_to_all_dists):
        q_d_inv = torch.ones(anchor_to_all_dists.shape)
        q_d_inv = q_d_inv / q_d_inv.sum()
        return q_d_inv.detach().cpu().numpy()

    def inverse_sphere_distances(
        self, dim, bs, anchor_to_all_dists, labels, anchor_label
    ):
        dists = anchor_to_all_dists

        # negated log-distribution of distances of unit sphere in dimension <dim>
        log_q_d_inv = (2.0 - float(dim)) * torch.log(dists) - (
            float(dim - 3) / 2
        ) * torch.log(1.0 - 0.2499 * (dists.pow(2)))
        log_q_d_inv[np.where(labels == anchor_label)[0]] = 0

        q_d_inv = torch.exp(
            log_q_d_inv - torch.max(log_q_d_inv)
        )  # - max(log) for stability
        q_d_inv[np.where(labels == anchor_label)[0]] = 0

        ### NOTE: Cutting of values with high distances made the results slightly worse. It can also lead to
        # errors where there are no available negatives (for high samples_per_class cases).
        # q_d_inv[np.where(dists.detach().cpu().numpy()>self.upper_cutoff)[0]]    = 0

        q_d_inv = q_d_inv / q_d_inv.sum()
        return q_d_inv.detach().cpu().numpy()

    def pdist(self, A):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min=0)
        res = 4 * res / res.max()  # normalization to max added by me
        return res.sqrt()


# from ete3 tutorial
def search_by_size(node, size):
    "Finds nodes with a given number of leaves"
    matches = []
    for n in node.traverse():
        if len(n) == size:
            if len(n.get_children()) >= size:
                matches.append(n)
    return matches


# modified from ete3 codebase
def my_convert_to_ultrametric(tree, tree_length=None, strategy="fixed_child"):
    """
    .. versionadded: 2.1

    Converts a tree into ultrametric topology (all leaves must have
    the same distance to root). Note that, for visual inspection
    of ultrametric trees, node.img_style["size"] should be set to
    0.
    """

    # Could something like this replace the old algorithm?
    # most_distant_leaf, tree_length = self.get_farthest_leaf()
    # for leaf in self:
    #    d = leaf.get_distance(self)
    #    leaf.dist += (tree_length - d)
    # return

    # get origin distance to root
    dist2root = {tree: 0.0}
    for node in tree.iter_descendants("levelorder"):
        dist2root[node] = dist2root[node.up] + node.dist

    # get tree length by the maximum
    if not tree_length:
        tree_length = max(dist2root.values())
    else:
        tree_length = float(tree_length)

    # converts such that starting from the leaves, each group which can have
    # the smallest step, does. This ensures all from the same genus are assumed the same
    # space apart
    if strategy == "fixed_child":
        step = 1.0

        # pre-calculate how many splits remain under each node
        node2max_depth = {}
        for node in tree.traverse("postorder"):
            if not node.is_leaf():
                max_depth = max([node2max_depth[c] for c in node.children]) + 1
                node2max_depth[node] = max_depth
            else:
                node2max_depth[node] = 1
        node2dist = {tree: 0.0}
        # modify the dist property of nodes
        for node in tree.iter_descendants("levelorder"):
            node.dist = tree_length - node2dist[node.up] - node2max_depth[node] * step

            # print(node,node.dist, node.up)
            node2dist[node] = node.dist + node2dist[node.up]

    return tree


class VanillaVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List = None,
        phylo_path=None,
        **kwargs
    ) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.phylo_path = phylo_path
        self.tree = Tree(self.phylo_path)
        self.species, self.T = self.calc_T()

        # self.alpha = nn.Parameter(torch.tensor(0.5))

        self.T = torch.tensor(self.T).cuda().float()
        self.T = self.T / self.T.max()
        self.inv_T = self.T.inverse()
        self.batchminer = BatchMiner(self.species, self.T)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # self.distance_model = nn.Sequential(
        #    nn.Linear(256, 128),
        #    nn.ReLU(),
        #    nn.Linear(128, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 32),
        #    nn.ReLU(),
        #    nn.Linear(32, 16),
        #    nn.ReLU(),
        #    nn.Linear(16, 8),
        #    nn.ReLU(),
        #    nn.Linear(8, 4),
        #    nn.ReLU(),
        #    nn.Linear(4, 2),
        #    nn.ReLU(),
        #    nn.Linear(2, 1),
        # )

    def calc_T(self):
        t = self.tree.copy()
        t = my_convert_to_ultrametric(t)
        T = np.zeros((215, 215))
        # First, assign mu to each of the leaves/species
        species = []
        i = 0
        for leaf1 in t.get_leaves():
            species += [leaf1.name]
            j = 0
            for leaf2 in t.get_leaves():
                ancestor = leaf1.get_common_ancestor(leaf2)
                T[i, j] = ancestor.get_distance(t)
                j += 1
            i += 1

        return species, T

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        labels = kwargs["labels"]

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        phylo_weight = kwargs["phylo_weight"]
        recons_weight = kwargs["recons_weight"]
        phylo_size = kwargs["phylo_size"]
        phylo_mean = kwargs["phylo_mean"]
        triplet_weight = kwargs["triplet_weight"]

        kld_loss = torch.mean(
            -0.5
            * torch.sum(
                1
                + log_var[:, phylo_size:]
                - mu[:, phylo_size:] ** 2
                - log_var[:, phylo_size:].exp(),
                dim=1,
            ),
            dim=0,
        )

        # Trying to put T into our Char KL divergence
        # cov_prior = torch.matmul(torch.matmul(mu.T,batch_T),torch.pinv(mu.T))
        # cov = torch.matmul(mu.T,mu)

        # kld_loss = 0.5*torch.mean(
        #    torch.trace(torch.log(cov_prior)) - torch.trace()
        # )

        # Using https://math.stackexchange.com/questions/3130131/log-det-a-tr-loga
        # and the fact that T is a positive symmetric matrix,
        # we get that log(det(T)) = trace(log(T))
        # Still not certain if the covariance of log_var is the same as the log of covariace of var

        # get current covariance between samples
        # Through some mathy math.... Hopefully it's correct

        # cov = (
        #    0.5
        #    * (
        #        log_var[:, :phylo_space_size].sum(dim=1).unsqueeze(0)
        #        + log_var[:, :phylo_space_size].sum(dim=1).unsqueeze(1)
        #    )
        # ).exp()

        # cov = torch.cov(log_var[:, :phylo_space_size])

        # mu_T_mu = torch.matmul(mu[:, :phylo_space_size].T, batch_inv_T)
        # mu_T_mu = torch.matmul(mu_T_mu, mu[:, :phylo_space_size])
        # mu_T_mu = torch.diag(mu_T_mu)

        # old but works
        # phylo_kld_loss = 0.5 * torch.mean(
        #    torch.trace(torch.log(batch_T))
        #    - torch.trace(cov)
        #    - 1
        #    + mu_T_mu
        #    + torch.trace(torch.matmul(batch_inv_T, cov))
        # )

        # phylo_loss = self.phylo_loss_function(mu, log_var, labels)

        # Using Felsensteins likelihood and kl divergence of two multivariate gaussians
        # We convert felsensteins matrix into character space from species space
        #   using the fact that Trace(X.T*A*X) = Trace(X*U*X.T) where U = X.T*A*X.T_pinv
        # Then if A was our inverse covariance matrix in species space, U is our inverse covariance matrix in character space

        # --- MOST RECENT PHYLO LOSS ---

        samples = self.reparameterize(mu, log_var)
        # cov_q = torch.cov(samples[:, :phylo_size]) # old covariance calculation

        # for this covariance calculation, each row is multiplied by each other row
        #   and then averaged together over the characters to produce a covariance between each of the samples
        cov_q = torch.einsum(
            "...j,ij", samples[:, :phylo_size], samples[:, :phylo_size]
        ) / (phylo_size - 1)

        # new attempt, where we treat the covariance of each character seperately
        # creates a matrix of shape (batch_size, batch_size, latent_size)
        # cov_q = torch.einsum(
        #    "k...,l...", samples[:, :phylo_size], samples[:, :phylo_size]
        # )
        # cov_q = torch.permute(cov_q, (1, 2, 0))

        # new attempt, where we treat the covariance of each character seperately
        # creates a matrix of shape (batch_size, batch_size, latent_size)
        # but also subtract the mean, so we are effectively getting the covariance....
        # cov_q = torch.einsum(
        #    "k...,l...", samples[:, :phylo_size], samples[:, :phylo_size]
        # )
        # cov_q = torch.permute(cov_q, (1, 2, 0))

        if phylo_mean:
            unique_labels, labels_count = np.unique(labels, return_counts=True)
            unique_labels_num = torch.tensor(list(range(len(unique_labels)))).cuda()
            labels_num = torch.tensor(
                [list(unique_labels).index(x) for x in labels]
            ).cuda()
            species_means = (
                torch.zeros((len(unique_labels), mu.shape[1]), dtype=torch.float)
                .cuda()
                .scatter_add_(
                    0, labels_num.unsqueeze(1).repeat(1, samples.shape[1]), samples
                )
            )
            species_means = species_means / torch.tensor(
                labels_count
            ).cuda().float().unsqueeze(1)

            cov_q = torch.einsum(
                "...j,ij", species_means[:, :phylo_size], species_means[:, :phylo_size]
            ) / (phylo_size - 1)

            batch_indices = [self.species.index(x) for x in unique_labels]
            batch_T = self.T[batch_indices][:, batch_indices]

            batch_inv_T = (
                batch_T.inverse()
            )  # self.inv_T[batch_indices][:, batch_indices]
            no_species = len(unique_labels)

            phylo_kld_loss = (
                0.5
                * (
                    torch.log(torch.linalg.eig(batch_T)[0].real).sum() / no_species
                    + -torch.log(torch.linalg.eig(cov_q)[0].real).sum() / no_species
                    # torch.log(torch.linalg.eig(cov_q)[0].real).sum()
                    # / no_species  # the log of the determinant is the log of the product of eigenvalues..
                    + -1
                    + torch.diag(
                        torch.matmul(
                            torch.matmul(species_means[:, :phylo_size].T, batch_inv_T),
                            species_means[:, :phylo_size],
                        )
                    )
                    / no_species
                    + torch.trace(torch.matmul(batch_inv_T, cov_q)) / no_species
                    # + torch.einsum("ii...", torch.matmul(batch_inv_T, cov_q))/ bs  # gets the trace per character
                ).mean()
            )

        else:
            batch_indices = [self.species.index(x) for x in labels]
            batch_T = self.T[batch_indices][:, batch_indices]
            batch_inv_T = self.inv_T[batch_indices][:, batch_indices]

            # batch_inv_T = torch.inverse(batch_T)
            eps = 1e-15
            bs = mu.shape[0]

            phylo_kld_loss = (
                0.5
                * (
                    torch.trace(torch.log(batch_T)) / bs
                    + -torch.einsum("ii", (torch.log(cov_q)))
                    / bs  # takes the trace along the first two axes
                    + -1
                    + torch.diag(
                        torch.matmul(
                            torch.matmul(mu[:, :phylo_size].T, batch_inv_T),
                            mu[:, :phylo_size],
                        )
                    )
                    / bs
                    + torch.einsum("ii", torch.matmul(batch_inv_T, cov_q)) / bs
                    # + torch.einsum("ii...", torch.matmul(batch_inv_T, cov_q))/ bs  # gets the trace per character
                ).mean()
            )

        # / --- MOST RECENT PHYLO LOSS ---

        # phylo_kld_loss = torch.mean((cov_q - #batch_T) ** 2)
        # species_dists_weight = 0.001
        # want the distances between all species to match the phylo tree
        # batch_dist_prior = 1 - batch_T
        # batch_dist_calcd = torch.cdist(samples[:, :20], samples[:, :20])
        # species_dists_loss = (
        #    (batch_dist_calcd - self.alpha * batch_dist_prior) ** 2
        # ).sum()

        # --- MOST RECENT TRIPLET LOSS ---

        # triplet_weight = 0.00005
        sampled_triplets = self.batchminer(samples[:, :phylo_size], labels)
        triplet_loss = torch.stack(
            [
                F.triplet_margin_loss(
                    samples[triplet[0], :phylo_size],
                    samples[triplet[1], :phylo_size],
                    samples[triplet[2], :phylo_size],
                    margin=triplet[3],
                )
                for triplet in sampled_triplets
            ]
        ).mean()

        # / --- MOST RECENT TRIPLET LOSS ---

        # dist_weight = 0.0001
        # intraspecies_dists = torch.scalar_tensor#(0).cuda()
        # for species_idx in np.unique#(batch_indices):
        #   species_mu = mu[np.where(np.array(batch_indices) == species_idx)][:63]
        #    species_mean = species_mu.mean()
        #    intraspecies_dists += ((species_mu - species_mean) ** 2).sum()

        # distance_weight = 1e-3
        # distance_loss = self.calc_distance_loss(mu, labels)

        loss = (
            recons_weight * recons_loss
            + phylo_weight * phylo_kld_loss
            + kld_weight * kld_loss
            # + distance_weight * distance_loss
            # + triplet_weight * triplet_loss
            # + species_dists_weight * species_dists_loss
        )  #  + phylo_weight * phylo_kld_loss

        if torch.isnan(phylo_kld_loss):
            wtf

        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": kld_loss.detach(),  # kl divergence between characters
            # "Dist": distance_loss.detach()
            "Phylo_Loss": phylo_kld_loss.detach(),
            # "Triplet": triplet_loss.detach(),
            # "Species_dists": species_dists_loss.detach(),
            # "Instraspecies_Loss": intraspecies_dists.detach(),
        }

    def calc_distance_loss(self, mu, labels):
        reference_dists = torch.tensor([]).cuda()
        pred_dists = torch.tensor([]).cuda()
        # for each pair (i,j) of inputs
        for i in range(len(mu)):
            batch_size = len(mu)
            mu_i = mu[i].unsqueeze(0).repeat(batch_size, 1)
            # mu_i is now mu of image i repeated batch_size number of times
            mu_i = torch.concat((mu_i, mu), axis=1)
            label_i = labels[i]
            label_i_idx = self.species.index(label_i)

            labels_indices = [self.species.index(x) for x in labels]
            ref_dists_i = 1 - self.T[label_i_idx, labels_indices]
            reference_dists = torch.concat((reference_dists, ref_dists_i))
            pred_dists = torch.concat(
                (pred_dists, self.distance_model(mu_i).reshape(-1))
            )

        return (pred_dists - reference_dists).abs().mean()

    def phylo_loss_function_eq_6(self, mu, log_var, labels):  # eequation 6
        samples = mu  # self.reparameterize(mu, log_var)

        batch_indices = [self.species.index(x) for x in labels]

        xi = samples[:, :2]
        batch_inv_T = self.inv_T[batch_indices][:, batch_indices]
        exponent = torch.matmul(xi.T, batch_inv_T)
        exponent = torch.matmul(exponent, xi)
        return exponent.sum()

    def phylo_loss_function_new(self, mu, labels):
        t = self.tree.copy()
        species = np.unique(labels)
        species = [t.get_leaves_by_name(name)[0] for name in species]

        # Remove any species from the tree that are not in this batch
        t.prune(species)
        t.resolve_polytomy()  # For algorithm to work, needs to be binary tree
        t.standardize()  # Removes any redundant nodes

        # First, assign mu to each of the leaves/species

        # Rob's Hack - Add a term for the distance between the specimens in this space
        h_D = []
        dists = torch.cdist(mu, mu) ** 2
        for leaf in t.get_leaves():
            sp = leaf.name
            leaf.species = [sp]
            batch_indices = np.where(np.array(labels) == sp)[0]
            _mu = mu[batch_indices].mean(axis=0)
            n = len(batch_indices)

            D = dists[batch_indices, batch_indices]
            leaf.x = _mu
            leaf.n = n
            h_D += [D.sum() / 2]

        # recursively go through the tree:
        # 1. calculate D_ab^2/(v_a + v_b) for each group of two sister leaves
        # 2. collapse those leaves, and set parent node mu to weighted mean of those mus
        # 3. repeat until calculated for whole tree

        g_Dv = []
        while len(t) > 1:
            # Get all nodes with two leaves
            for node in search_by_size(t, size=2):
                l = node.get_leaves()

                i0 = np.isin(np.array(labels), l[0].species)
                i1 = np.isin(np.array(labels), l[1].species)

                dists_01 = dists[i0][:, i1]

                u = dists_01.mean()
                v0, v1 = l[0].dist, l[1].dist

                # Get average mu
                # Calculated based on Felsenstein paper,
                # but has problem that if v0 or v1 are 0 (if you remove polytomies), removes part of the distance equation
                # node.x = (x0*v1 + x1*v0)/(v0+v1)

                node.species = l[0].species + l[1].species

                g_Dv += [[node.name, u / (v0 + v1)]]
                node.remove_child(l[0])
                node.remove_child(l[1])
        return sum(h_D) + sum([x[1] for x in g_Dv])

    def phylo_loss_function_new_split(self, mu, labels):  # _new_split
        t = self.tree.copy()
        species = np.unique(labels)
        species = [t.get_leaves_by_name(name)[0] for name in species]

        # Remove any species from the tree that are not in this batch
        t.prune(species)
        t.resolve_polytomy()  # For algorithm to work, needs to be binary tree
        t.standardize()  # Removes any redundant nodes

        # Use only half of mu
        split_mu = mu[:, :63]

        # First, assign mu to each of the leaves/species

        # Rob's Hack - Add a term for the distance between the specimens in this space
        h_D = []
        dists = torch.cdist(split_mu, split_mu) ** 2
        for leaf in t.get_leaves():
            sp = leaf.name
            leaf.species = [sp]
            batch_indices = np.where(np.array(labels) == sp)[0]
            _mu = split_mu[batch_indices].mean(axis=0)
            n = len(batch_indices)

            D = dists[batch_indices, batch_indices]
            leaf.x = _mu
            leaf.n = n
            h_D += [D.sum() / 2]

        # recursively go through the tree:
        # 1. calculate D_ab^2/(v_a + v_b) for each group of two sister leaves
        # 2. collapse those leaves, and set parent node mu to weighted mean of those mus
        # 3. repeat until calculated for whole tree

        g_Dv = []
        while len(t) > 1:
            # Get all nodes with two leaves
            for node in search_by_size(t, size=2):
                l = node.get_leaves()

                i0 = np.isin(np.array(labels), l[0].species)
                i1 = np.isin(np.array(labels), l[1].species)

                dists_01 = dists[i0][:, i1]

                u = dists_01.mean()
                v0, v1 = l[0].dist, l[1].dist

                # Get average mu
                # Calculated based on Felsenstein paper,
                # but has problem that if v0 or v1 are 0 (if you remove polytomies), removes part of the distance equation
                # node.x = (x0*v1 + x1*v0)/(v0+v1)

                node.species = l[0].species + l[1].species

                g_Dv += [[node.species, u / (v0 + v1)]]
                node.remove_child(l[0])
                node.remove_child(l[1])
        return sum(h_D) + sum([x[1] for x in g_Dv])

    def phylo_loss_function(self, mu, log_var, labels):  # old split
        samples = self.reparameterize(mu, log_var)

        t = self.tree.copy()
        species = np.unique(labels)
        species = [t.get_leaves_by_name(name)[0] for name in species]

        # Remove any species from the tree that are not in this batch
        t.prune(species)

        # Use only half of mu
        split_samples = samples[:, :96]

        # Move specimens up a level
        for leaf in t.get_leaves():
            sp = leaf.name
            batch_indices = np.where(np.array(labels) == sp)[0]
            n = len(batch_indices)
            _mu = split_samples[batch_indices]
            for i in range(n):
                new_leaf = Tree(name=str(i))
                new_leaf.x = _mu[i]
                new_leaf.n = 1
                leaf.add_child(child=new_leaf, dist=1)

        t = my_convert_to_ultrametric(t)
        t.resolve_polytomy()  # For algorithm to work, needs to be binary tree
        t.standardize()

        # First, assign mu to each of the leaves/species

        # Rob's Hack - Add a term for the distance between the specimens in this space
        # h_D = []
        # dists = torch.cdist(split_samples, split_samples)
        # for leaf in t.get_leaves():
        #    sp = leaf.name
        #    batch_indices = np.where(np.array(labels) == sp)[0]
        #    _mu = split_samples[batch_indices].mean(axis=0)
        #    n = len(batch_indices)

        #    D = dists[batch_indices, batch_indices]
        #    leaf.x = _mu  # Is this valid? taking the mean of the samples? should instead do this by sample probably
        #    leaf.n = n
        #    h_D += [(D**2).sum() / 2]

        # recursively go through the tree:
        # 1. calculate D_ab^2/(v_a + v_b) for each group of two sister leaves
        # 2. collapse those leaves, and set parent node mu to weighted mean of those mus
        # 3. repeat until calculated for whole tree

        g_Dv = []
        while len(t) > 1:
            # Get all nodes with two leaves
            for node in search_by_size(t, size=2):
                l = node.get_leaves()

                v0, v1 = l[0].dist, l[1].dist
                x0, x1 = l[0].x, l[1].x
                # n0, n1 = l[0].n, l[1].n

                # Get average mu
                # Calculated based on Felsenstein paper,
                # but has problem that if v0 or v1 are 0 (if you remove polytomies), removes part of the distance equation
                # node.x = (x0*v1 + x1*v0)/(v0+v1)

                node.x = x0 * v1 / (v0 + v1) + x1 * v0 / (v0 + v1)
                if not len(node.name):
                    node.name = l[0].name + ", " + l[1].name
                    node.species = list(l[0].name) + list(l[1].name)
                # node.n = n0 + n1
                node.dist += (v0 * v1) / (v0 + v1)  # Felsenstein's trick to avoid 0s

                # Get Distance
                D = ((x0 - x1) ** 2).sum()

                g_Dv += [[node.name, D / (v0 + v1)]]
                node.remove_child(l[0])
                node.remove_child(l[1])
        g_Dv = torch.tensor([x[1] for x in g_Dv])

        return g_Dv.sum() + ((g_Dv.mean() - 10) ** 2)

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
