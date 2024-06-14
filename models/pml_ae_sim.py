import copy
import itertools

import os
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
from itertools import combinations, permutations

from torch.nn import Sequential


class BatchMiner():
    def __init__(self):
        pass

    def __call__(self, batch, labels, hierarchical=False):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()



        # get a batch_size number of samples from same class to different class
        # get a batch_size number of samples from all different classes

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



class HierarchicalBatchMiner():
    def __init__(self, distances_labels, distances, distance_multiplier=5):
        self.name = "hierarchical"
        self.species = distances_labels
        #self.dists = distances
        self.dists = distances
        self.distance_multiplier = distance_multiplier

    def __call__(self, batch, labels, hierarchical=False):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()



        # get a batch_size number of samples from same class to different class
        # get a batch_size number of samples from all different classes

        unique_classes = np.unique(labels)
        indices        = np.arange(len(batch))
        # class dict is a dictionary of {class_name: list of batch samples in that class}
        class_dict     = {i:indices[np.array(labels)==i] for i in unique_classes}

        # class triplets is now a list of all the possible triplet combinations based on classes
        class_triplets = [list(itertools.product([x],[x],[y for y in unique_classes if x!=y])) for x in unique_classes]
        class_triplets = [list(x) for y in class_triplets for x in y]
        # add the distance margin
        if hierarchical:
            class_triplets = [x + [self.distance_multiplier*self.dists[self.species.index(x[0]),self.species.index(x[-1])]] for x in class_triplets]



        sampled_triplets = [[x for x in list(itertools.product(*[class_dict[j] for j in i[:3]] + [[i[3]]])) if x[0]!=x[1]] for i in class_triplets]
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


class PMLAESIM(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        n_characters: int,
        seq_length: int,
        encoder_arch = [836*5, 100],
        decoder_arch = [100, 836*5],
        phylo_path=None,
        triplet_margin=0.1,
        hierarchical=False,
        distance_multiplier=5,
        **kwargs,
    ) -> None:
        super(PMLAESIM, self).__init__()
        self.activation_function = "ReLU"
        self.enable_bn = False
        self.seq_length = seq_length
        self.margin = triplet_margin

        self.encoder_arch = encoder_arch
        self.decoder_arch = decoder_arch

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
        self.hierarchical=hierarchical
        self.distance_multiplier = distance_multiplier

        self.n_characters = n_characters
        self.phylo_path = phylo_path
        self.tree = Tree(self.phylo_path)
        self.T_labels, self.T = self.calc_T()
        self.distances_labels, self.distances = self.calc_distances()
        
        if self.hierarchical:
            self.triplet_batchminer = HierarchicalBatchMiner(self.distances_labels, self.distances, self.distance_multiplier)
        else:
            self.triplet_batchminer = BatchMiner()

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        mu = self.encoder(input.squeeze().reshape(-1, 5 * self.seq_length))


        return mu


    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        bs, n_classes = z.shape

        X = self.decoder(z).reshape(bs, self.seq_length, 5)

        result = F.softmax(X, dim=2)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:

        mu = self.encode(input)

        return [self.decode(mu), input.squeeze(), mu, 0]



    def triplet_loss(self, samples, labels, margin):
        sampled_triplets = self.triplet_batchminer(samples, labels)
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

    def hier_triplet_loss(self, samples, labels, class_indices, hierarchical=True):
        sampled_triplets = self.triplet_batchminer(samples, labels, hierarchical)
        if len(sampled_triplets) < 2:
            return torch.tensor(0)
        triplet_loss = torch.stack(
            [
                F.triplet_margin_loss(
                    samples[triplet[0]],
                    samples[triplet[1]],
                    samples[triplet[2]],
                    margin=triplet[3],
                )
                for triplet in sampled_triplets
            ]
        ).mean()
        return triplet_loss

    def blomberg_loss(self, samples, labels):
        unique_labels = list(np.unique(labels))
        # map the labels into integers

        int_labels = torch.tensor([unique_labels.index(x) for x in labels])
        int_labels = int_labels.view(int_labels.size(0), 1).expand(-1, samples.size(1))

        unique_int_labels, labels_count = int_labels.unique(dim=0, return_counts=True)

        avg_per_label = torch.zeros_like(unique_int_labels.cuda(), dtype=torch.float).scatter_add_(0, int_labels.cuda(), samples)
        avg_per_label = avg_per_label / labels_count.float().unsqueeze(1).cuda()
        # avg_per_label, in order of unique_labels



        sub_T_labels = [x for x in self.T_labels if x in unique_labels]
        batch_label_indices = [self.T_labels.index(x) for x in sub_T_labels]
        avg_per_label = avg_per_label[[unique_labels.index(x) for x in sub_T_labels]]
        
        V = torch.tensor(self.T).cuda()[batch_label_indices,:][:,batch_label_indices].float()
        V_inv = V.inverse().float()
        n = len(batch_label_indices) # number of species

        a_hat = torch.matmul(V_inv, avg_per_label).sum(axis=0)/V_inv.sum()
        MSE_0 = ((avg_per_label - a_hat)**2).sum(axis=0) / (n-1)
        MSE = torch.matmul(torch.matmul((avg_per_label - a_hat).T,V_inv),(avg_per_label - a_hat)).diag() / (n-1)
        observed_ratio = MSE_0 / MSE
        expected_ratio = (1/(n-1))*(torch.diag(V).sum() - n / V_inv.sum())
        return ((observed_ratio - expected_ratio)**2).sum()

    def blomberg_p_loss(self, samples, labels):
        unique_labels = list(np.unique(labels))
        # map the labels into integers

        int_labels = torch.tensor([unique_labels.index(x) for x in labels])
        int_labels = int_labels.view(int_labels.size(0), 1).expand(-1, samples.size(1))

        unique_int_labels, labels_count = int_labels.unique(dim=0, return_counts=True)

        avg_per_label = torch.zeros_like(unique_int_labels.cuda(), dtype=torch.float).scatter_add_(0, int_labels.cuda(), samples)
        avg_per_label = avg_per_label / labels_count.float().unsqueeze(1).cuda()
        # avg_per_label, in order of unique_labels



        sub_T_labels = [x for x in self.T_labels if x in unique_labels]
        batch_label_indices = [self.T_labels.index(x) for x in sub_T_labels]
        avg_per_label = avg_per_label[[unique_labels.index(x) for x in sub_T_labels]]


        V = torch.tensor(self.T).cuda()[batch_label_indices,:][:,batch_label_indices].float()
        V_inv = V.inverse().float()
        n = len(batch_label_indices) # number of species

        a_hat = torch.matmul(V_inv, avg_per_label).sum(axis=0)/V_inv.sum()
        MSE_0 = ((avg_per_label - a_hat)**2).sum(axis=0) / (n-1)
        MSE = torch.matmul(torch.matmul((avg_per_label - a_hat).T,V_inv),(avg_per_label - a_hat)).diag() / (n-1)
        observed_ratio = MSE_0 / MSE
        expected_ratio = (1/(n-1))*(torch.diag(V).sum() - n / V_inv.sum())

        # we want to minimize the p value though, so we want the traits when rearranged to have a lower K 
        # than the one observed.
        shuffled_indices = np.arange(len(avg_per_label))
        np.random.shuffle(shuffled_indices)
        avg_per_label_rearranged = avg_per_label[shuffled_indices]
        a_hat = torch.matmul(V_inv, avg_per_label_rearranged).sum(axis=0)/V_inv.sum()
        MSE_0 = ((avg_per_label_rearranged - a_hat)**2).sum(axis=0) / (n-1)
        MSE = torch.matmul(torch.matmul((avg_per_label_rearranged - a_hat).T,V_inv),(avg_per_label_rearranged - a_hat)).diag() / (n-1)
        shuffled_observed_ratio = MSE_0 / MSE

        return ((observed_ratio - expected_ratio)**2).mean() + torch.maximum(shuffled_observed_ratio - observed_ratio, torch.tensor(0)).mean()

    def blomberg_mse_loss(self, samples, labels):
        # --- Get Average trait per label --- 
        # get unique labels, this rearranges them alphabetically
        unique_labels = list(np.unique(labels))
        # map the labels into integers

        int_labels = torch.tensor([unique_labels.index(x) for x in labels])
        int_labels = int_labels.view(int_labels.size(0), 1).expand(-1, samples.size(1))

        unique_int_labels, labels_count = int_labels.unique(dim=0, return_counts=True)

        avg_per_label = torch.zeros_like(unique_int_labels.cuda(), dtype=torch.float).scatter_add_(0, int_labels.cuda(), samples)
        avg_per_label = avg_per_label / labels_count.float().unsqueeze(1).cuda()
        # avg_per_label, in order of unique_labels



        sub_T_labels = [x for x in self.T_labels if x in unique_labels]
        batch_label_indices = [self.T_labels.index(x) for x in sub_T_labels]
        avg_per_label = avg_per_label[[unique_labels.index(x) for x in sub_T_labels]]


        # code to check that this worked properly
        # import pandas as pd
        # df = pd.DataFrame(samples.cpu().detach().numpy(), index=labels)
        # df = df.reset_index().groupby('index').mean()
        # df[sub_T_labels]


        # Get covariance for only those species that are included in this batch
        V = torch.tensor(self.T).cuda()[batch_label_indices,:][:,batch_label_indices].float()
        V = V - V.min() # since we are using a subset, the root might move.
        V_inv = V.inverse().float()
        n = len(batch_label_indices) # number of species

        a_hat = torch.matmul(V_inv, avg_per_label).sum(axis=0)/V_inv.sum()
        X = avg_per_label

        MSE = torch.matmul(torch.matmul((X - a_hat).T,V_inv),(X - a_hat)).diag() / (n-1)

        return MSE.sum()

    def blomberg_mse_kld_loss(self, samples, labels):
        unique_labels = list(np.unique(labels))
        assert len(labels) == 2*len(unique_labels), "Currently only supports two labels per class"
        n_traits = samples.shape[-1]
        labels = np.array(labels)

        batch_label_indices = [self.T_labels.index(x) for x in unique_labels]
        V = torch.tensor(self.T).cuda()[batch_label_indices,:][:,batch_label_indices].float()
        V_inv = V.inverse().float()

        blomberg_kld = torch.tensor(0).cuda().float()
        traits_included = 1
        for trait in range(n_traits):
            indices = np.argsort(labels)
            labels = labels[indices]
            X = samples[indices,trait]
            X = X.reshape(-1, 2)
            # now X is a nx2 matrix, where n is number of species, 2 examples per species

            avg = X.mean(axis=1)
            n = len(avg)
            a_hat = torch.matmul(V_inv, avg).sum(axis=0)/V_inv.sum()

            cov_q = torch.outer((avg - a_hat),(avg-a_hat)) 

            MSE = torch.matmul(torch.matmul((avg - a_hat).T,V_inv),(avg - a_hat)) / (n-1)
            b = 0.5 * (
                torch.log(torch.linalg.eigvalsh(V)).sum()
                - torch.log(torch.linalg.eigvalsh(cov_q)).sum() #  + (torch.eye(n)*1e-5).cuda())
                - (n-1)
                + MSE*(n-1)
                + torch.trace(torch.matmul(V_inv, cov_q))
            )
            if not torch.isnan(b):
                blomberg_kld += b
                traits_included += 1
            else:
                print('nan')

        if torch.isnan(blomberg_kld):
            wtf
        return blomberg_kld / traits_included


    def blomberg_mse_kld_tot_loss(self, samples, labels):
        unique_labels = list(np.unique(labels))
        assert len(labels) == 2*len(unique_labels), "Currently only supports two labels per class"
        n_traits = samples.shape[-1]
        labels = np.array(labels)

        sub_T_labels = [x for x in self.T_labels if x in unique_labels]
        batch_label_indices = [self.T_labels.index(x) for x in sub_T_labels]
        V = torch.tensor(self.T).cuda()[batch_label_indices,:][:,batch_label_indices].float()
        V_inv = V.inverse().float()

        indices = np.argsort([sub_T_labels.index(x) for x in labels])

        labels = labels[indices]
        X = samples[indices]
        X = X.reshape(-1, 2, n_traits)
        # now X is a nx2 matrix, where n is number of species, 2 examples per species

        avg = X.mean(axis=1)
        n = avg.shape[0] # number of species
        a_hat = torch.matmul(V_inv, avg).sum(axis=0)/V_inv.sum()

        cov_q = torch.matmul((avg - a_hat),(avg-a_hat).T) / n_traits

        MSE = torch.matmul(torch.matmul((avg - a_hat).T,V_inv),(avg - a_hat)).diag() / (n-1)
        blomberg_kld = 0.5 * (
            torch.log(torch.linalg.eigvalsh(V)).sum()
            - torch.log(torch.linalg.eigvalsh(cov_q + (torch.eye(n)*1e-5).cuda())).sum() #  
            - (n-1)
            + MSE*(n-1)
            + torch.trace(torch.matmul(V_inv, cov_q))
        ).mean()
        if torch.isnan(blomberg_kld):
            print('nan')
            blomberg_kld = torch.tensor(0).cuda().float()

        return blomberg_kld




    def cov_mse_loss(self, samples, labels):


        # --- Get Average trait per label --- 
        # get unique labels, this rearranges them alphabetically
        unique_labels = list(np.unique(labels))
        # map the labels into integers

        int_labels = torch.tensor([unique_labels.index(x) for x in labels])
        int_labels = int_labels.view(int_labels.size(0), 1).expand(-1, samples.size(1))

        unique_int_labels, labels_count = int_labels.unique(dim=0, return_counts=True)

        avg_per_label = torch.zeros_like(unique_int_labels.cuda(), dtype=torch.float).scatter_add_(0, int_labels.cuda(), samples)
        avg_per_label = avg_per_label / labels_count.float().unsqueeze(1).cuda()
        # avg_per_label, in order of unique_labels



        sub_T_labels = [x for x in self.T_labels if x in unique_labels]
        batch_label_indices = [self.T_labels.index(x) for x in sub_T_labels]
        avg_per_label = avg_per_label[[unique_labels.index(x) for x in sub_T_labels]]


        # code to check that this worked properly
        # import pandas as pd
        # df = pd.DataFrame(samples.cpu().detach().numpy(), index=labels)
        # df = df.reset_index().groupby('index').mean()
        # df[sub_T_labels]

        X = avg_per_label
        n_characters = X.shape[1]
        n_classes = X.shape[0]

        indices = [self.T_labels.index(x) for x in sub_T_labels]
        T = torch.tensor(self.T).cuda()[indices][:, indices] 
        T = T - T.min() # since we are using a subset, the root might move. 
        inv_T = T.inverse().float()

        a_hat = torch.matmul(inv_T, X).sum(axis=0)/inv_T.sum()
        _X = X - a_hat
        cov_q = torch.matmul(_X, _X.T) / (n_classes - 1)

        kld_loss = ((cov_q - T)**2).mean()

        if torch.isnan(kld_loss):
            wtf
        return kld_loss




    def likelihood_loss(self, samples, labels):


        # --- Get Average trait per label --- 
        # get unique labels, this rearranges them alphabetically
        unique_labels = list(np.unique(labels))
        # map the labels into integers

        int_labels = torch.tensor([unique_labels.index(x) for x in labels])
        int_labels = int_labels.view(int_labels.size(0), 1).expand(-1, samples.size(1))

        unique_int_labels, labels_count = int_labels.unique(dim=0, return_counts=True)

        avg_per_label = torch.zeros_like(unique_int_labels.cuda(), dtype=torch.float).scatter_add_(0, int_labels.cuda(), samples)
        avg_per_label = avg_per_label / labels_count.float().unsqueeze(1).cuda()
        # avg_per_label, in order of unique_labels



        sub_T_labels = [x for x in self.T_labels if x in unique_labels]
        batch_label_indices = [self.T_labels.index(x) for x in sub_T_labels]
        avg_per_label = avg_per_label[[unique_labels.index(x) for x in sub_T_labels]]


        # code to check that this worked properly
        # import pandas as pd
        # df = pd.DataFrame(samples.cpu().detach().numpy(), index=labels)
        # df = df.reset_index().groupby('index').mean()
        # df[sub_T_labels]

        X = avg_per_label
        n_characters = X.shape[1]
        n_classes = X.shape[0]

        indices = [self.T_labels.index(x) for x in sub_T_labels]
        V = torch.tensor(self.T).cuda()[indices][:, indices] 
        #T = T - T.min() # since we are using a subset, the root might move. 
        inv_V = V.inverse().float()



        a_hat = torch.matmul(inv_V, X).sum(axis=0)/inv_V.sum()
        _X = X - a_hat

        #variance = torch.matmul(torch.matmul(_X.T,inv_T), _X).diag()/(n_classes-1)
        variance = torch.var(_X, dim=0, correction=1)
        N = n_classes
        L = n_characters

        likelihood = 0.5*N*L*torch.log(torch.tensor(2*torch.pi)) \
        + 0.5*L*torch.log(torch.det(V)) \
        + N*torch.log(torch.sqrt(variance)).sum() \
        + 0.5*torch.dot(variance, torch.diag(torch.matmul(torch.matmul(_X.T,inv_V),_X)))
         
        if torch.isnan(likelihood):
            wtf
        return likelihood



    def mse_variance_loss(self, samples, labels):
        # --- Get Average trait per label --- 
        # get unique labels, this rearranges them alphabetically
        unique_labels = list(np.unique(labels))
        # map the labels into integers

        int_labels = torch.tensor([unique_labels.index(x) for x in labels])
        int_labels = int_labels.view(int_labels.size(0), 1).expand(-1, samples.size(1))

        unique_int_labels, labels_count = int_labels.unique(dim=0, return_counts=True)

        avg_per_label = torch.zeros_like(unique_int_labels.cuda(), dtype=torch.float).scatter_add_(0, int_labels.cuda(), samples)
        avg_per_label = avg_per_label / labels_count.float().unsqueeze(1).cuda()
        # avg_per_label, in order of unique_labels



        sub_T_labels = [x for x in self.T_labels if x in unique_labels]
        batch_label_indices = [self.T_labels.index(x) for x in sub_T_labels]
        avg_per_label = avg_per_label[[unique_labels.index(x) for x in sub_T_labels]]


        # code to check that this worked properly
        # import pandas as pd
        # df = pd.DataFrame(samples.cpu().detach().numpy(), index=labels)
        # df = df.reset_index().groupby('index').mean()
        # df[sub_T_labels]

        X = avg_per_label
        n_characters = X.shape[1]
        n_classes = X.shape[0]

        indices = [self.T_labels.index(x) for x in sub_T_labels]
        V = torch.tensor(self.T).cuda()[indices][:, indices] 
        inv_V = V.inverse().float()


        a_hat = torch.matmul(inv_V, X).sum(axis=0)/inv_V.sum()
        _X = X - a_hat

        variance = torch.var(_X, dim=0, correction=1)

        return ((variance - 1)**2).mean()


    def mse_covariance_loss(self, samples, labels):
        # --- Get Average trait per label --- 
        # get unique labels, this rearranges them alphabetically
        unique_labels = list(np.unique(labels))
        # map the labels into integers

        int_labels = torch.tensor([unique_labels.index(x) for x in labels])
        int_labels = int_labels.view(int_labels.size(0), 1).expand(-1, samples.size(1))

        unique_int_labels, labels_count = int_labels.unique(dim=0, return_counts=True)

        avg_per_label = torch.zeros_like(unique_int_labels.cuda(), dtype=torch.float).scatter_add_(0, int_labels.cuda(), samples)
        avg_per_label = avg_per_label / labels_count.float().unsqueeze(1).cuda()
        # avg_per_label, in order of unique_labels



        sub_T_labels = [x for x in self.T_labels if x in unique_labels]
        batch_label_indices = [self.T_labels.index(x) for x in sub_T_labels]
        avg_per_label = avg_per_label[[unique_labels.index(x) for x in sub_T_labels]]


        # code to check that this worked properly
        # import pandas as pd
        # df = pd.DataFrame(samples.cpu().detach().numpy(), index=labels)
        # df = df.reset_index().groupby('index').mean()
        # df[sub_T_labels]

        X = avg_per_label
        n_characters = X.shape[1]
        n_classes = X.shape[0]

        indices = [self.T_labels.index(x) for x in sub_T_labels]
        V = torch.tensor(self.T).cuda()[indices][:, indices] 
        inv_V = V.inverse().float()


        a_hat = torch.matmul(inv_V, X).sum(axis=0)/inv_V.sum()
        _X = X - a_hat

        covariance = torch.matmul(_X, _X.T)/n_characters

        return ((covariance - V)**2).mean()




    def mse_mean_loss(self, samples, labels):
        # --- Get Average trait per label --- 
        # get unique labels, this rearranges them alphabetically
        unique_labels = list(np.unique(labels))
        # map the labels into integers

        int_labels = torch.tensor([unique_labels.index(x) for x in labels])
        int_labels = int_labels.view(int_labels.size(0), 1).expand(-1, samples.size(1))

        unique_int_labels, labels_count = int_labels.unique(dim=0, return_counts=True)

        avg_per_label = torch.zeros_like(unique_int_labels.cuda(), dtype=torch.float).scatter_add_(0, int_labels.cuda(), samples)
        avg_per_label = avg_per_label / labels_count.float().unsqueeze(1).cuda()
        # avg_per_label, in order of unique_labels



        sub_T_labels = [x for x in self.T_labels if x in unique_labels]
        batch_label_indices = [self.T_labels.index(x) for x in sub_T_labels]
        avg_per_label = avg_per_label[[unique_labels.index(x) for x in sub_T_labels]]


        # code to check that this worked properly
        # import pandas as pd
        # df = pd.DataFrame(samples.cpu().detach().numpy(), index=labels)
        # df = df.reset_index().groupby('index').mean()
        # df[sub_T_labels]

        X = avg_per_label
        n_characters = X.shape[1]
        n_classes = X.shape[0]

        indices = [self.T_labels.index(x) for x in sub_T_labels]
        V = torch.tensor(self.T).cuda()[indices][:, indices] 
        inv_V = V.inverse().float()


        a_hat = torch.matmul(inv_V, X).sum(axis=0)/inv_V.sum()

        return ((a_hat)**2).mean()


    def likelihood_loss_0_1(self, samples, labels):


        # --- Get Average trait per label --- 
        # get unique labels, this rearranges them alphabetically
        unique_labels = list(np.unique(labels))
        # map the labels into integers

        int_labels = torch.tensor([unique_labels.index(x) for x in labels])
        int_labels = int_labels.view(int_labels.size(0), 1).expand(-1, samples.size(1))

        unique_int_labels, labels_count = int_labels.unique(dim=0, return_counts=True)

        avg_per_label = torch.zeros_like(unique_int_labels.cuda(), dtype=torch.float).scatter_add_(0, int_labels.cuda(), samples)
        avg_per_label = avg_per_label / labels_count.float().unsqueeze(1).cuda()
        # avg_per_label, in order of unique_labels



        sub_T_labels = [x for x in self.T_labels if x in unique_labels]
        batch_label_indices = [self.T_labels.index(x) for x in sub_T_labels]
        avg_per_label = avg_per_label[[unique_labels.index(x) for x in sub_T_labels]]


        # code to check that this worked properly
        # import pandas as pd
        # df = pd.DataFrame(samples.cpu().detach().numpy(), index=labels)
        # df = df.reset_index().groupby('index').mean()
        # df[sub_T_labels]

        X = avg_per_label
        n_characters = X.shape[1]
        n_classes = X.shape[0]

        indices = [self.T_labels.index(x) for x in sub_T_labels]
        V = torch.tensor(self.T).cuda()[indices][:, indices] 
        #T = T - T.min() # since we are using a subset, the root might move. 
        inv_V = V.inverse().float()



        a_hat = torch.matmul(inv_V, X).sum(axis=0)/inv_V.sum()
        _X = X - a_hat

        #variance = torch.matmul(torch.matmul(_X.T,inv_T), _X).diag()/(n_classes-1)
        variance = torch.ones(_X.shape[1]).cuda()#torch.var(_X, dim=0, correction=1)
        N = n_classes
        L = n_characters

        likelihood = 0.5*N*L*torch.log(torch.tensor(2*torch.pi)) \
        + 0.5*L*torch.log(torch.det(V)) \
        + N*torch.log(torch.sqrt(variance)).sum() \
        + 0.5*torch.dot(variance, torch.diag(torch.matmul(torch.matmul(X.T,inv_V),X)))
         
        if torch.isnan(likelihood):
            wtf
        return likelihood






    def distance_loss(self, samples, labels):
        # from felsenstein, page 402, D^2 = vt*p
        # where vt is the branch length from species 1 to 2


        # --- Get Average trait per label --- 
        # get unique labels, this rearranges them alphabetically
        unique_labels = list(np.unique(labels))
        # map the labels into integers

        int_labels = torch.tensor([unique_labels.index(x) for x in labels])
        int_labels = int_labels.view(int_labels.size(0), 1).expand(-1, samples.size(1))

        unique_int_labels, labels_count = int_labels.unique(dim=0, return_counts=True)

        avg_per_label = torch.zeros_like(unique_int_labels.cuda(), dtype=torch.float).scatter_add_(0, int_labels.cuda(), samples)
        avg_per_label = avg_per_label / labels_count.float().unsqueeze(1).cuda()
        # avg_per_label, in order of unique_labels



        sub_Vt_labels = [x for x in self.distances_labels if x in unique_labels]
        batch_label_indices = [self.distances_labels.index(x) for x in sub_Vt_labels]
        avg_per_label = avg_per_label[[unique_labels.index(x) for x in sub_Vt_labels]]


        # code to check that this worked properly
        # import pandas as pd
        # df = pd.DataFrame(samples.cpu().detach().numpy(), index=labels)
        # df = df.reset_index().groupby('index').mean()
        # df[sub_T_labels]

        X = avg_per_label
        n_characters = X.shape[1]
        n_classes = X.shape[0]

        indices = [self.distances_labels.index(x) for x in sub_Vt_labels]
        Vt = torch.tensor(self.distances).cuda()[indices][:, indices] 

        loss = torch.tensor(0).cuda().float()
        for i in range(n_classes):

            sigma = X.std(axis=0)
            for j in range(i+1, n_classes):
                Xi = avg_per_label[i]
                Xj = avg_per_label[j]
                d2 = ((Xi-Xj)**2/sigma**2).mean() # outer subtraction
                weight = 1#/Vt[i,j]
                loss += weight*((Vt[i,j] - d2)**2)


        if torch.isnan(loss):
            wtf
        return loss

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        inp = args[1]
        n_classes = inp.shape[-1]
        mu = args[2]
        labels = kwargs["labels"]
        losses = kwargs["losses"]



        training = kwargs["training"]
        if training:
            optimizers = kwargs["optimizers"]
            model_opt = optimizers
            backward = kwargs["backward"]


        loss = torch.tensor(0).cuda().float()
        logs = {}

        if training:
            model_opt.zero_grad()

        for weight, loss_name in losses:
            if loss_name == "kld":
                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
                kld_loss = torch.mean(kld_loss)
                loss += weight * kld_loss
                logs["KLD"] = kld_loss.detach()
            elif loss_name == "triplet":
                #samples = self.reparameterize(mu, log_var)
                triplet_loss = self.triplet_loss(mu, labels, self.margin)
                loss += weight * triplet_loss
                logs["Triplet"] = triplet_loss.detach()
            elif loss_name == "hier_triplet":
                assert self.hierarchical == True
                class_indices = [i for i in range(len(self.distances_labels)) if self.distances_labels[i] in labels]
                triplet_loss = self.triplet_loss(
                    mu, labels, class_indices, hierarchical=True
                )
                loss += weight * triplet_loss
                logs["Triplet"] = triplet_loss.detach()
            elif loss_name == "recons":
                recons_loss = F.mse_loss(recons, inp) # , reduction="sum"
                loss += weight * recons_loss
                logs["Reconstruction_Loss"] = recons_loss.detach()
            elif loss_name == "blomberg":
                blomberg_loss = self.blomberg_loss(mu, labels)
                loss += weight * blomberg_loss
                logs["Blomberg_Loss"] = blomberg_loss.detach()
            elif loss_name == "blomberg_p":
                blomberg_loss = self.blomberg_p_loss(mu, labels)
                loss += weight * blomberg_loss
                logs["Blomberg_P_Loss"] = blomberg_loss.detach()
            elif loss_name == "blomberg_mse":
                blomberg_loss = self.blomberg_mse_loss(mu, labels)
                loss += weight * blomberg_loss
                logs["Blomberg_MSE_Loss"] = blomberg_loss.detach()
            elif loss_name == "blomberg_mse_kld":
                blomberg_kld_loss = self.blomberg_mse_kld_loss(mu, labels)
                loss += weight * blomberg_kld_loss
                logs["Blomberg_MSE_KLD_Loss"] = blomberg_kld_loss.detach()
            elif loss_name == "blomberg_mse_kld_tot":
                blomberg_kld_loss = self.blomberg_mse_kld_tot_loss(mu, labels)
                loss += weight * blomberg_kld_loss
                logs["Blomberg_MSE_KLD_Loss"] = blomberg_kld_loss.detach()
            elif loss_name == "cov_mse":
                cov_mse_loss = self.cov_mse_loss(mu, labels)
                loss += weight * cov_mse_loss
                logs["Cov_MSE_Loss"] = cov_mse_loss.detach()
            elif loss_name == "likelihood":
                likelihood_loss = self.likelihood_loss(mu, labels)
                loss += weight * likelihood_loss
                logs["Likelihood Loss"] = likelihood_loss.detach()
            elif loss_name == "likelihood_0_1":
                likelihood_loss = self.likelihood_loss_0_1(mu, labels)
                loss += weight * likelihood_loss
                logs["Likelihood Loss 01"] = likelihood_loss.detach()
            elif loss_name == "mse_covariance":
                likelihood_loss = self.mse_covariance_loss(mu, labels)
                loss += weight * likelihood_loss
                logs["MSE Covariance"] = likelihood_loss.detach()
            elif loss_name == "mse_variance":
                likelihood_loss = self.mse_variance_loss(mu, labels)
                loss += weight * likelihood_loss
                logs["MSE Variance"] = likelihood_loss.detach()
            elif loss_name == "mse_mean":
                likelihood_loss = self.mse_mean_loss(mu, labels)
                loss += weight * likelihood_loss
                logs["MSE Mean"] = likelihood_loss.detach()
            elif loss_name == "distance":
                distance_loss = self.distance_loss(mu, labels)
                loss += weight * distance_loss
                logs["Distance Loss"] = distance_loss.detach()
            else:
                wtf


        logs["loss"] = loss

        if torch.isnan(loss):
            wtf
            
        if training:
            backward(loss)
            model_opt.step()

        return logs

    def calc_distances(self):
        t = self.tree.copy()
        no_species = len(self.tree.get_leaf_names())
        D = np.zeros((no_species, no_species))
        # First, assign mu to each of the leaves/species
        species = []

        leaves = t.get_leaves()
        leaves = sorted(leaves, key=lambda x: x.name)
        for i in tqdm(range(len(leaves))):
            leaf1 = leaves[i]
            species += [leaf1.name]
            j = 0
            for j in range(i, len(leaves)):
                leaf2 = leaves[j]
                D[i, j] = leaf1.get_distance(leaf2)
                D[j, i] = D[i, j]

        return species, D


    def calc_T(self):
        t = self.tree.copy()
        no_species = len(self.tree.get_leaf_names())
        T = np.zeros((no_species, no_species))
        # First, assign mu to each of the leaves/species
        species = []

        leaves = t.get_leaves()
        #leaves = sorted(leaves, key=lambda x: x.name)
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
        tree = matrix_to_tree(mu, labels)
        _gt.prune([x.name for x in tree.get_leaves() if x.name in _gt.get_leaf_names()])
        tree.prune([x.name for x in _gt.get_leaves() if x.name in tree.get_leaf_names()])
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
        print("Initialized Encoder: %s" % self)

    def forward(self, x):
        x = self.net(x)
        z_mu = self.l_mu(x)
        return z_mu


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