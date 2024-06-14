import copy
import itertools

import torch
import random
import numpy as np
from ete3 import Tree
from models import BaseVAE
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from .types_ import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from munkres import Munkres
from scipy.optimize import linear_sum_assignment
from itertools import combinations

from torch.nn import Sequential




class BatchMiner():
    def __init__(self):
        pass

    def __call__(self, batch, labels):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        unique_classes = np.unique(labels)
        indices        = np.arange(len(batch))
        class_dict     = {i:indices[np.array(labels)==i] for i in unique_classes}

        sampled_triplets = [list(itertools.product([x],[x],[y for y in unique_classes if x!=y])) for x in unique_classes]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        sampled_triplets = [[x for x in list(itertools.product(*[class_dict[j] for j in i])) if x[0]!=x[1]] for i in sampled_triplets]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        no_samples = min(batch.shape[0], len(sampled_triplets))
        #NOTE: The number of possible triplets is given by #unique_classes*(2*(samples_per_class-1)!)*(#unique_classes-1)*samples_per_class
        sampled_triplets = random.sample(sampled_triplets, no_samples)
        return sampled_triplets
# Code modified from: Mary K. Kuhner, Jon Yamato, Practical Performance of Tree Comparison Metrics, Systematic Biology, Volume 64, Issue 2, March 2015, Pages 205–214, https://doi.org/10.1093/sysbio/syu085
# which was provided under public domain license
def ars(set1, set2):
    # intersection over union
    numer = len(set1.intersection(set2))
    denom = len(set1.union(set2))
    assert denom > 0  # don't divide by zero!
    return float(numer) / float(denom)

def treealign(intree1, intree2, assign_align_scores_to_nodes=False):
    tree1 = copy.deepcopy(intree1)
    tree2 = copy.deepcopy(intree2)

    i = 0
    for node1 in tree1.iter_descendants("postorder"):
        if not node1.is_leaf():
            node1.name = i
            i += 1

    j = 0
    for node2 in tree2.iter_descendants("postorder"):
        if not node2.is_leaf():
            node2.name = j
            j += 1

    n = max(i, j)
    vals = np.zeros((i, j))

    allnodes = set(tree1.get_leaf_names())
    allnodes_check = set(tree2.get_leaf_names())
    assert allnodes == allnodes_check, (
        "all nodes tree1:"
        + str(allnodes)
        + ",\n all nodes tree2: "
        + str(allnodes_check)
    )

    for node1 in tree1.iter_descendants("postorder"):
        if node1.is_leaf():
            continue
        i0 = set(node1.get_leaf_names())
        i1 = allnodes.difference(i0)
        for node2 in tree2.iter_descendants("postorder"):
            if node2.is_leaf():
                continue
            j0 = set(node2.get_leaf_names())
            j1 = allnodes.difference(j0)
            a00 = ars(i0, j0)
            a11 = ars(i1, j1)
            a01 = ars(i0, j1)
            a10 = ars(i1, j0)
            s = max(min(a00, a11), min(a01, a10))
            vals[node1.name][node2.name] = 1.0 - s
        m = Munkres()

    total = 0
    col_to_align_score = {}
    row_idxs = []
    for row, column in zip(*linear_sum_assignment(vals)):
        value = vals[row][column]
        total += value
        col_to_align_score[column] = str(value) + "_" + str(row) + "-" + str(column)
        row_idxs += [row]

    for node2 in tree2.iter_descendants("postorder"):
        if node2.name in list(col_to_align_score.keys()):
            node2.name = str(col_to_align_score[node2.name])
        elif len(str(node2.name)) > 3:
            node2.name = node2.name.replace(" ", "_")
        else:
            node2.name = "?"

    if assign_align_scores_to_nodes:
        return total, tree1, tree2, vals
    else:
        return total


# from https://github.com/scipy/scipy/issues/8274
def get_newick(node, newick, parentdist, leaf_names):
    """
    Converts scipy tree to newick format
    """
    if node.is_leaf():
        return "%s:%.2f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parentdist - node.dist, newick)
        else:
            newick = ");"
        newick = get_newick(node.get_left(), newick, node.dist, leaf_names)
        newick = get_newick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
        newick = "(%s" % (newick)
        return newick


def matrix_to_tree(mat, names):
    Z = hierarchy.linkage(mat, "single")

    tree = hierarchy.to_tree(Z, False)

    newick = get_newick(tree, "", tree.dist, names)
    tree = Tree(newick)
    return tree


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
        node2dist = {tree: -1.0}
        # modify the dist property of nodes
        for node in tree.iter_descendants("levelorder"):
            node.dist = tree_length - node2dist[node.up] - node2max_depth[node] * step

            # print(node,node.dist, node.up)
            node2dist[node] = node.dist + node2dist[node.up]

    return tree


class PMLVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        n_characters: int,
        triplet_margin: 0.1,
        hidden_dims: List = None,
        phylo_path=None,
        **kwargs,
    ) -> None:
        super(PMLVAE, self).__init__()
        self.activation_function = "ReLU"
        self.enable_bn = False

        self.encoder_arch= [836 * 5, 1000]#[836 * 5, 1500, 1500]
        self.decoder_arch= [1000, 836 * 5]#[100, 2000, 836 * 5]

        self.encoder = Encoder(
            arch=self.encoder_arch,
            n_latent=n_characters,
            enable_bn=self.enable_bn,
            activation_function=self.activation_function,
        )
        self.decoder = Decoder(
            arch=self.decoder_arch,
            n_latent=n_characters,
            enable_bn=self.enable_bn,
            activation_function=self.activation_function,
        )


        self.n_characters = n_characters
        self.phylo_path = phylo_path
        self.tree = Tree(self.phylo_path)
        self.species, self.T = self.calc_T()
        self.margin = triplet_margin

        # self.alpha = nn.Parameter(torch.tensor(0.5))
        self.ordered_genera = self.get_ordered_genera()
        self.reordered_indices = [self.species.index(x) for x in self.ordered_genera]

        self.T = torch.tensor(self.T).cuda().float()
        self.T = self.T / self.T.max()
        self.T = self.reorder_indices(self.T)
        self.V = copy.deepcopy(self.T)
        self.v_tree = self.tree.copy()
        self.species = self.get_ordered_genera()

        self.batchminer = BatchMiner()

        self.inv_T = self.T.inverse()

    def triplet_loss(self, samples, labels, margin):
        sampled_triplets = self.batchminer(samples, labels)
        if len(sampled_triplets) < 2:
            return torch.tensor(0)
        triplet_loss = torch.stack(
            [
                F.triplet_margin_loss(
                    samples[triplet[0]],
                    samples[triplet[1]],
                    samples[triplet[2]],
                    margin=margin,
                )
                for triplet in sampled_triplets
            ]
        ).mean()
        return triplet_loss


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
        mu, logvar = self.encoder(input.squeeze().reshape(-1, 5 * 836))


        return mu, logvar, logvar


    def decode(self, z: Tensor, indices) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        bs, n_classes = z.shape

        X = self.decoder(z).reshape(bs, 836, 5)

        result = F.softmax(X, dim=2)
        return result

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var, indices = self.encode(input)
        # _X = mu - mu.mean(dim=1).unsqueeze(1)
        # cov_q = torch.matmul(_X.transpose(2, 1), _X)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z, indices), input.squeeze(), mu, log_var]



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
        n_classes = input.shape[-1]
        mu = args[2]
        log_var = args[3]
        labels = kwargs["labels"]
        losses = kwargs["losses"]


        def my_matmul(A, B):
            return torch.einsum("...ik,...jk", A, B)

        loss = torch.tensor(0).cuda().float()
        logs = {}


        for weight, loss_name in losses:
            if loss_name == "kld":
                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss += weight * kld_loss
                logs["KLD"] = kld_loss.detach()
            elif loss_name == "triplet":
                #samples = self.reparameterize(mu, log_var)
                triplet_loss = self.triplet_loss(mu, labels, self.margin)
                loss += weight * triplet_loss
                logs["Triplet"] = triplet_loss.detach()
            elif loss_name == "recons":
                recons_loss = F.mse_loss(recons, input, reduction="sum")
                loss += weight * recons_loss
                logs["Reconstruction_Loss"] = recons_loss.detach()
            else:
                wtf


        logs["loss"] = loss

        if torch.isnan(loss):
            wtf

        return logs

    def trace(self, X):
        return torch.einsum("...ii", X)



    def calc_RF_and_align(self, mu, labels, gt):
        """_summary_

        Args:
            cov_q (np.array): matrix of size (bs, n_genera, n_genera)
            labels (list): names of genera
            gt (Tree): Ground Truth Phylogeny in ete3 format
        """

        #whiten the data
        #U, s, Vt = np.linalg.svd(mu.cpu().detach().numpy(), full_matrices=False)
        #X_white = np.dot(U, Vt)

        _gt = gt.copy()
        for leaf in _gt.get_leaves():
            leaf.name = leaf.name.capitalize()
        tree = matrix_to_tree(mu, labels)
        _gt.prune([x.name for x in tree.get_leaves()])
        align = treealign(_gt, tree)
        rf, max_rf, _, _, _, _, _ = _gt.robinson_foulds(tree, unrooted_trees=True)
        return rf / max_rf, align

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




class Encoder(nn.Module):
    def __init__(
        self,
        arch = [263 * 23, 1500, 1500],
        n_latent=2,
        enable_bn=True,
        activation_function="relu",
    ):

        super(Encoder, self).__init__()

        exec(f"self.af = nn.{activation_function}")

        layers = []
        for i in range(len(arch[:-1])):
            _in, _out = arch[i], arch[i + 1]
            layers += [nn.Linear(in_features=_in, out_features=_out)]
            if enable_bn:
                layers += [nn.BatchNorm1d(_out)]
            layers += [self.af()]

        self.net = Sequential(*layers)

        self.l_mu = nn.Linear(in_features=arch[-1], out_features=n_latent)
        self.l_logsigma = nn.Linear(in_features=arch[-1], out_features=n_latent)
        print("Initialized Encoder: %s" % self)

    def forward(self, x):
        x = self.net(x)
        z_mu = self.l_mu(x)
        z_logsigma = self.l_logsigma(x)
        return z_mu, z_logsigma


class Decoder(nn.Module):
    def __init__(
        self,
        arch=[100, 500, 263 * 23],
        n_latent=2,
        enable_bn=True,
        activation_function="ReLU",
    ):

        super(Decoder, self).__init__()

        exec(f"self.af = nn.{activation_function}")
        arch = [n_latent] + arch
        self.layers = []
        for i in range(len(arch[:-1])):
            _in, _out = arch[i], arch[i + 1]
            if i != 0:
                self.layers += [self.af()]
            if enable_bn:
                self.layers += [nn.BatchNorm1d(_in)]
            self.layers += [nn.Linear(in_features=_in, out_features=_out)]

        #self.layers += [nn.Sigmoid()]
        self.net = Sequential(*self.layers)

        print("Initialized Decoder: %s" % self.net)

    def forward(self, x):
        x = self.net(x)

        return x