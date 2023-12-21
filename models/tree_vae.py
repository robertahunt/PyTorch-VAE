import torch
import numpy as np
from ete3 import Tree
from models import BaseVAE
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from .types_ import *
import seaborn as sns
import matplotlib.pyplot as plt


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


class TreeVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List = None,
        phylo_path=None,
        **kwargs,
    ) -> None:
        super(TreeVAE, self).__init__()

        self.latent_dim = latent_dim
        self.phylo_path = phylo_path
        self.tree = Tree(self.phylo_path)
        self.species, self.T = self.calc_T()

        # self.alpha = nn.Parameter(torch.tensor(0.5))
        self.ordered_genera = self.get_ordered_genera()
        self.reordered_indices = [self.species.index(x) for x in self.ordered_genera]

        self.T = torch.tensor(self.T).cuda().float()
        # self.T = self.T / self.T.max()
        self.T = self.reorder_indices(self.T)
        self.species = self.get_ordered_genera()

        self.inv_T = self.T.inverse()

        modules = []
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        # Build Encoder
        for i in range(len(hidden_dims)):
            h_dim = hidden_dims[i]
            stride = 1 if i == 0 else 6
            mpool = 3 if i != 2 else 2
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=(6, 1),
                        stride=(stride, 1),
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    nn.MaxPool2d((mpool, 1)),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(16 * 144, latent_dim * 6)
        # self.fc_var = nn.Linear(16 * 144, (latent_dim) ** 2)
        self.fc_var = nn.Linear(16 * 144, int((latent_dim**2 + latent_dim) / 2))

        self.decoder_input = nn.Linear(latent_dim, 16 * 144)

        hidden_dims.reverse()
        hidden_dims += [5]

        # Build Decoder
        modules = []

        kernel_sizes = [4, 18, 6]
        strides = [1, 18, 3]
        padding = [0, 7, 2]
        dilation = [3, 1, 1]
        for i in range(len(hidden_dims) - 1):
            h_dim = hidden_dims[i]
            stride = strides[i]
            kernel_size = kernel_sizes[i]

            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=(kernel_size, 1),
                        padding=0,
                        stride=(stride, 1),
                        output_padding=(padding[i], 0),
                        dilation=dilation[i],
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

    def reorder_indices(self, A):
        return A[np.ix_(self.reordered_indices, self.reordered_indices)]

    def heatmap(self, cov):
        plt.figure()
        sns.heatmap(cov)

    def get_ordered_genera(self):
        # fmt: off
        return ['Xylosandrus', 'Aeolus', 'Pyropyga', 'Carpophilus', 'Liriomyza',
       'Phytomyza', 'Botanophila', 'Dilophus', 'Bezzimyia', 'Prodiplosis',
       'Campylomyza', 'Culicoides', 'Forcipomyia', 'Chironomus',
       'Dicrotendipes', 'Microtendipes', 'Micropsectra', 'Paratanytarsus',
       'Rheotanytarsus', 'Tanytarsus', 'Allocladius', 'Chaetocladius',
       'Cricotopus', 'Diplocladius', 'Halocladius', 'Limnophyes',
       'Metriocnemus', 'Paraphaenocladius', 'Smittia', 'Ablabesmyia',
       'ChiroGn1', 'chiroJanzen01', 'Achradocera', 'Medetera',
       'Drosophila', 'Hirtodrosophila', 'Porphyrochroa', 'Rhamphomyia',
       'Fannia', 'Heleomyza', 'Platypalpus', 'Trichina', 'Ormosia',
       'Atherigona', 'Coenosia', 'Neodexiopsis', 'Musca', 'Macgrathphora',
       'Megaselia', 'Lutzomyia', 'Psychoda', 'Macrostenomyia', 'Scatopse',
       'Bradysia', 'Camptochaeta', 'Corynoptera', 'Cosmosciara',
       'Cratyna', 'Lycoriella', 'Peyerimhoffia', 'Pseudosciara',
       'Scatopsciara', 'Themira', 'Simulium', 'Bifronsina', 'Chespiritos',
       'Leptocera', 'Pseudocollinella', 'Pullimosina', 'Rachispoda',
       'Rudolfina', 'Ischiolepta', 'Hermetia', 'Phytomyptera',
       'Paradidyma', 'Trichocera', 'Physiphora', 'Aleurotrachelus',
       'Bemisia', 'Trialeurodes', 'Rhopalosiphum', 'Euceraphis',
       'Empoasca', 'Dinotrema', 'alyMalaise01', 'Blacus', 'Diospilus',
       'Chelonus', 'Ecphylus', 'Heterospilus', 'Meteorus', 'Leiophron',
       'Hormius', 'Apanteles', 'Cotesia', 'Diolcogaster',
       'Dolichogenidea', 'Glyptapanteles', 'Pseudapanteles',
       'mgMalaise160', 'Pambolus', 'encyrMalaise01', 'Palmistichus',
       'euloMalaise01', 'Procryptocerus', 'ichneuMalaise01', 'Hypsicera',
       'Chilocyrtus', 'Orthocentrus', 'Plectiscus', 'Stenomacrus',
       'Tersilochus', 'tersiMalaise01', 'Anagrus', 'Glyphidocera',
       'Taygete', 'blastoBioLep01', 'cosmoBioLep01', 'Cosmopterix',
       'cosmoMalaise01', 'Argyria', 'cramMalaise75', 'Antaeotricha',
       'Battaristis', 'Aristotelia', 'Dichomeris', 'Tuta', 'Sinoe',
       'Telphusa', 'gelBioLep01', 'gelBioLep1', 'gelMalaise01',
       'geleBioLep01', 'Glyphipterix', 'malaiseGraci01',
       'malaiseGraci01 Malaise4560', 'Stigmella', 'Idioglossa',
       'oecoMalaise01', 'Plutella', 'phyMalaise01', 'tinBioLep01',
       'Lepidopsocus', 'Thrips']
        # fmt: on

    def calc_T(self):
        t = self.tree.copy()
        t = my_convert_to_ultrametric(t)
        no_species = len(self.tree.get_leaf_names())
        T = np.zeros((no_species, no_species))
        # First, assign mu to each of the leaves/species
        species = []

        debug = False
        if debug:
            # don't bother calculating T since it takes so damn long for large phylogenies
            return self.tree.get_leaf_names(), np.diagflat(np.ones(no_species))

        leaves = t.get_leaves()
        leaves = sorted(leaves, key=lambda x: x.name)
        for i in tqdm(range(len(leaves))):
            leaf1 = leaves[i]
            species += [leaf1.name]
            j = 0
            for j in range(i, len(leaves)):
                leaf2 = leaves[j]
                ancestor = leaf1.get_common_ancestor(leaf2)
                T[i, j] = ancestor.get_distance(t)
                T[j, i] = T[i, j]

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
        mu = self.fc_mu(result).view(-1, 6, 144)

        A = self.fc_var(result)
        A = F.relu(A) + 1e-10
        bs = A.shape[0]
        U = torch.zeros((bs, 144, 144)).cuda()
        indices = torch.triu_indices(144, 144)
        U[:, indices[0], indices[1]] = A  # set upper matrix
        # U[:, indices[1], indices[0]] = A  # set upper matrix

        # logA = logA.view(-1, 144, 144)

        # To make sure std is a positive definite matrix,
        # multiply it by it's transpose, add a small eps to the diagonal,
        # then convert back to log....

        # following wikipedia (coming from Computation Statistics textbook, page 315/316), temporarily just try using A directly
        #    https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
        # std = torch.exp(0.5 * logvar)  # 0.5 since std**2 = var
        # std = torch.matmul(std, std.transpose(2, 1))
        # std.add_(torch.eye(144).cuda() * 0.1)
        # logvar = 2 * torch.log(std)

        # if torch.linalg.eigvals(torch.exp(0.5 * logvar)).real.min() < 0:
        #    wtf

        return [mu, U]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 16, 6, 144)
        result = self.decoder(result)
        result = F.softmax(result, dim=1)
        return result

    def reparameterize(self, mu: Tensor, U: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        # std = logA
        # heavily based on https://juanitorduz.github.io/multivariate_normal/

        # L = torch.linalg.cholesky(logvar)
        # Now torch.matmul(L[0],L[0].transpose(0,1)) should be very close to std_hat[0]

        eps = torch.randn_like(mu)

        # multiply each of the matrices for each in batch, with vector of eps

        # samples = mu + torch.einsum("...jk,...ik", L, eps)
        samples = mu + torch.einsum("...jk,...ik", U, eps)

        return samples

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
        input = input.view(-1, 5, 836, 144)
        mu = args[2]
        U = args[3]
        labels = kwargs["labels"]

        # std = logA  # 0.5 since std**2 = var
        cov_q = torch.matmul(
            U, U.transpose(2, 1)
        )  # torch.matmul(std, std.transpose(2, 1))
        logvar = torch.log(cov_q)

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        def trace(X):
            return torch.einsum("...ii", X)

        def my_matmul(A, B):
            return torch.einsum("...ik,...jk", A, B)

        kld_loss = 0.5 * torch.mean(
            torch.mean(
                # +trace(torch.log(self.T)) # constant
                -trace(logvar).unsqueeze(1)
                + torch.diagonal(
                    my_matmul(my_matmul(mu, self.inv_T), mu), dim1=1, dim2=2
                )
                + trace(torch.matmul(self.inv_T, cov_q)).unsqueeze(1)
                - 144,
                dim=1,
            ),
            dim=0,
        )

        loss = recons_loss + kld_weight * kld_loss

        if torch.isnan(loss):
            wtf

        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": kld_loss.detach(),  # kl divergence between characters
        }

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
