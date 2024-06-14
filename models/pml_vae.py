import copy
import itertools

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
from scipy.cluster import hierarchy
from munkres import Munkres
from scipy.optimize import linear_sum_assignment
from itertools import combinations


class BatchMiner:
    def __init__(self, species, T):
        self.lower_cutoff = 0.5
        self.upper_cutoff = 1.4
        self.name = "distance"
        self.species = species
        self.dists = 0.5*(1 - T / T.max())
        self.combinations = np.array(list(combinations(range(16), 2)))
        self.n_triplets = 10

    def make_pos_neg_pairs(self, class_indices=None):
        if class_indices is None:
            class_indices = list(range(len(self.dists)))
        pos_neg_pairs = {}
        sub_dists = self.dists[class_indices][:, class_indices]
        _max = sub_dists.max()
        for a in range(len(class_indices)):
            pos_neg_pairs[a] = []
            ps = torch.where(sub_dists[a] < _max)[0]
            for p in ps:
                ns = torch.where(sub_dists[a] > sub_dists[a, p])[0]
                for n in ns:
                    m = sub_dists[a, n] - sub_dists[a, p]
                    pos_neg_pairs[a] += [(p, n, m)]
        return pos_neg_pairs

    def __call__(
        self,
        batch,
        labels,
        class_indices,
        hierarchical=False,
        tar_labels=None,
        return_distances=False,
        distances=None,
    ):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(labels, list):
            labels = np.array(labels)
        bs, n_chars, n_classes = batch.shape
        n_pairs_per_sample = n_classes #

        sub_dists = self.dists[class_indices][:, class_indices]
        # pos_neg = self.make_pos_neg_pairs(class_indices=class_indices)
        indices = np.random.choice(len(self.combinations), 16)
        range_class_indices = range(len(class_indices))

        triplets = []
        # use one sample from the batch as the anchor and take the positives and negatives from another
        for a, o in self.combinations[indices]:
            anchors, others = batch[a], batch[o]

            # for anchor_idx in range(n_classes):
            #    pos_neg_anchor = torch.tensor(pos_neg[anchor_idx])
            k = 0
            while k < n_pairs_per_sample:
                a, b,c = np.random.choice(range_class_indices, 3)
                # get 3 random classes, all different

                if hierarchical:
                    b_margin = sub_dists[a,b]
                    c_margin = sub_dists[a,c]
                    if b_margin == c_margin:
                        continue
                    if b_margin < c_margin:
                        pos = others[:,b]
                        neg = others[:,c]
                        margin = c_margin - b_margin
                    else:
                        pos = others[:,c]
                        neg = others[:,b]
                        margin = b_margin - c_margin


                    anchor = anchors[:,a]
                    triplets += [[anchor, pos, neg, margin]]
                    k += 1
                else:
                    c_margin = 0.5

                pos = others[:,a]
                neg = others[:,c]
                triplets += [[anchor, pos, neg, c_margin]]
                k += 1


        return triplets

    def equal_distances(self, anchor_to_all_dists):
        q_d_inv = torch.ones(anchor_to_all_dists.shape)
        q_d_inv = q_d_inv / q_d_inv.sum()
        return q_d_inv.detach().cpu().numpy()

    def pdist(self, A):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min=0)
        res = 4 * res / res.max()  # normalization to max added by me
        return res.sqrt()


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


class TreeVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        n_characters: int,
        hidden_dims: List = None,
        phylo_path=None,
        **kwargs,
    ) -> None:
        super(TreeVAE, self).__init__()

        self.n_characters = n_characters
        self.phylo_path = phylo_path
        self.tree = Tree(self.phylo_path)
        self.species, self.T = self.calc_T()

        # self.alpha = nn.Parameter(torch.tensor(0.5))
        self.ordered_genera = self.get_ordered_genera()
        self.reordered_indices = [self.species.index(x) for x in self.ordered_genera]

        self.T = torch.tensor(self.T).cuda().float()
        self.T = self.T / self.T.max()
        self.T = self.reorder_indices(self.T)
        self.V = copy.deepcopy(self.T)
        self.v_tree = self.tree.copy()
        self.species = self.get_ordered_genera()

        self.batchminer = BatchMiner(self.species, self.T)

        self.inv_T = self.T.inverse()

        # fmt: off
        # size (bs*144, 5, 836)
        self.enc1 = nn.Sequential( nn.Conv1d(5, out_channels=64, kernel_size=6, stride=1,), nn.BatchNorm1d(64), nn.LeakyReLU())
        # size: (bs*144, 64, 831)
        self.enc_mp1 = nn.MaxPool1d(3, return_indices=True)
        # size after: (bs*144, 64, 277)
        self.enc2 = nn.Sequential( nn.Conv1d(64, out_channels=32, kernel_size=6, stride=3,), nn.BatchNorm1d(32), nn.LeakyReLU())
        # size: (bs*144, 32, 91)
        self.enc_mp2 = nn.MaxPool1d(3, return_indices=True)
        # size after: (bs*144, 32, 30)
        self.enc3 = nn.Sequential( nn.Conv1d(32, out_channels=16, kernel_size=6, stride=3,), nn.BatchNorm1d(16), nn.LeakyReLU())
        # size: (bs*144, 16, 9)
        self.enc_mp3 = nn.MaxPool1d(3, return_indices=True)
        # size after: (bs*144, 16, 3)
        self.enc4 = nn.Sequential( nn.Conv1d(16, out_channels=self.n_characters*2, kernel_size=3, stride=1,))
        # size after: (bs*144, 6, 1)
        
        # size (bs*144, 6, 1)
        self.dec1 = nn.Sequential( nn.ConvTranspose1d(self.n_characters, out_channels=16, kernel_size=3, stride=1,), nn.BatchNorm1d(16), nn.LeakyReLU())
        self.dec_mp1 = nn.ConvTranspose1d(16, out_channels=16, kernel_size=3, stride=3,)#nn.MaxUnpool1d(3)
        # size (bs*144, 16, 6)
        self.dec2 = nn.Sequential( nn.ConvTranspose1d(16, out_channels=32, kernel_size=6, stride=3,), nn.BatchNorm1d(32), nn.LeakyReLU(), )
        self.dec_mp2 = nn.ConvTranspose1d(32, out_channels=32, kernel_size=3, stride=3, output_padding=1)#nn.MaxUnpool1d(3, padding=-1)
        # size (bs*144, 32, 30)
        self.dec3 = nn.Sequential( nn.ConvTranspose1d(32, out_channels=64, kernel_size=6, stride=3, output_padding=1), nn.BatchNorm1d(64), nn.LeakyReLU(), )
        self.dec_mp3 = nn.ConvTranspose1d(64, out_channels=64, kernel_size=3, stride=3,)#nn.MaxUnpool1d(3)
        # size (bs*144, 64, 277)
        self.dec4 = nn.Sequential( nn.ConvTranspose1d(64, out_channels=5, kernel_size=6, stride=1))
        # size (bs*144, 5, 836)
        # fmt: on

        # self.dec_mp1_indices = self.make_indices(3, 16, 3).cuda()
        # self.dec_mp2_indices = self.make_indices(3, 32, 30).cuda()
        # self.dec_mp3_indices = self.make_indices(3, 64, 277).cuda()

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

    def calc_V(self, X, labels):
        """
        Calculates the covariance matrix parameters based on the traits
        :param X: BxMxN matrix where B is the batch size,  N is the number of species/classes, M is the number of characters
        :param labels: N labels of the species / groupings
        :return:
        """
        X = X.mean(axis=0).T
        t = self.tree.copy()
        t = my_convert_to_ultrametric(t)
        N = len(labels)
        V = np.zeros((N, N))
        # First, assign mu to each of the leaves/species
        species = []

        # 1. Assign an average x to each leaf
        # 2. Go through the parents of the leaves, get the average of the xs as the x vector
        #    and calculate the v for all of its children
        # 3. Go through all the parents of the parents and calculate the v for all their children
        # 4. Go through all nodes from top down and calculate the v distance from the root to that node
        # 5. for each pair of leaves, find the common ancestor and get the v distance - this is the covariance.

        t.prune(labels)
        leaves = t.get_leaves()
        # sort the leaves so the index matches X
        leaves = sorted(leaves, key=lambda x: labels.index(x.name))

        # clean tree - remove all nodes with only one child
        for node in t.traverse("postorder"):
            if len(node.get_children()) == 1:
                node.delete()

        for i in range(len(leaves)):
            leaf = leaves[i]
            leaf.x = X[i]

        for node in t.traverse("postorder"):
            if node.is_leaf():
                continue

            children = node.get_children()
            children_x = torch.vstack([child.x for child in children])
            node.x = children_x.mean(axis=0)
            # to do: check if it makes sense that this is divided by the length of children.
            node.children_v = (torch.pdist(children_x)**2).mean() / len(children)

        t.ancestral_v = 0
        for node in t.traverse("preorder"):
            if len(node.get_ancestors()) == 0:
                continue
            parent = node.get_ancestors()[0]
            node.ancestral_v = parent.ancestral_v + parent.children_v
            if not node.is_leaf():
                node.support = node.ancestral_v

        for i in range(len(leaves)):
            leaf1 = leaves[i]
            species += [leaf1.name]
            j = 0
            for j in range(i, len(leaves)):
                leaf2 = leaves[j]
                ancestor = leaf1.get_common_ancestor(leaf2)
                V[i, j] = ancestor.ancestral_v
                V[j, i] = V[i, j]

        return species, torch.tensor(V).cuda().float()
    def update_V(self, X, labels):
        """
        Calculates the covariance matrix parameters based on the traits
        :param X: BxMxN matrix where B is the batch size,  N is the number of species/classes, M is the number of characters
        :param labels: N labels of the species / groupings
        :return:
        """

        with torch.no_grad():
            labels = [x[0] for x in labels]
            X = X.transpose(dim0=1,dim1=2)
            t = self.v_tree
            t = my_convert_to_ultrametric(t)
            N = len(labels)
            V = np.zeros((N, N))
            # First, assign mu to each of the leaves/species
            species = []

            # 1. Assign an average x to each leaf
            # 2. Go through the parents of the leaves, get the average of the xs as the x vector
            #    and calculate the v for all of its children
            # 3. Go through all the parents of the parents and calculate the v for all their children
            # 4. Go through all nodes from top down and calculate the v distance from the root to that node
            # 5. for each pair of leaves, find the common ancestor and get the v distance - this is the covariance.

            t.prune(labels)
            leaves = t.get_leaves()
            # sort the leaves so the index matches X
            leaves = sorted(leaves, key=lambda x: labels.index(x.name))

            # clean tree - remove all nodes with only one child
            for node in t.traverse("postorder"):
                if len(node.get_children()) == 1:
                    node.delete()

            for i in range(len(leaves)):
                leaf = leaves[i]
                leaf.x = X[:,i].mean(axis=0)
                parent = leaf.get_ancestors()[0]
                leaf.t = leaf.get_distance(parent)
                leaf._t = leaf.t


            t.t = 0
            for node in t.traverse("postorder"):
                if node.is_leaf():
                    continue

                children = node.get_children()
                ancestors = node.get_ancestors()
                if len(ancestors) != 0:
                    parent = ancestors[0]
                    node.t = node.get_distance(parent)


                total_t = sum([1/child._t for child in children])
                children_x = torch.stack([(1/child._t) * child.x / total_t for child in children])
                node.x = children_x.sum(axis=0)
                node._t = node.t + children_x.prod(axis=0) / children_x.sum(axis=0)
                node.dist = node._t.mean()

            #sigmas = []
            #for node in t.traverse("postorder"):
            #    if node.is_leaf():
            #        continue

            #    children = node.get_children()
            #    # get all possible pairs of children
            #    pairs = itertools.permutations(children, 2)
            #    for x1, x2 in pairs:
            #        sigma_squared = (x1.x - x2.x).var()/(x1.t + x2.t)
            #        sigmas += [sigma_squared]
            #avg_sigma_squared = torch.tensor(sigmas).mean()


            #t.v = 0
            #for node in t.traverse("postorder"):
            #    if len(node.get_ancestors()) == 0:
            #        continue
            #    parent = node.get_ancestors()[0]
            #    node.t = ((node.x.mean(axis=0) - parent.x.mean(axis=0))**2).mean() / avg_sigma_squared
            #    node.v = node.t * avg_sigma_squared
            #    node.dist = node.t

            t.ancestral_v = 0
            for node in t.traverse("preorder"):
                if len(node.get_ancestors()) == 0:
                    continue
                parent = node.get_ancestors()[0]
                node.ancestral_v = parent.ancestral_v + parent.dist


            for i in range(len(leaves)):
                leaf1 = leaves[i]
                species += [leaf1.name]
                j = 0
                for j in range(i, len(leaves)):
                    leaf2 = leaves[j]
                    ancestor = leaf1.get_common_ancestor(leaf2)
                    V[i, j] = ancestor.ancestral_v
                    V[j, i] = V[i, j]

            self.V = torch.tensor(V).cuda().float() / V.max()

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
        n_classes = input.shape[-1]

        # mu_var = self.encoder(input.permute(0, 3, 1, 2).reshape(-1, 5, 836))

        X = self.enc1(input.permute(0, 3, 1, 2).reshape(-1, 5, 836))
        X, ind1 = self.enc_mp1(X)
        X = self.enc2(X)
        X, ind2 = self.enc_mp2(X)
        X = self.enc3(X)
        X, ind3 = self.enc_mp3(X)
        mu_var = self.enc4(X)

        indices = [ind1, ind2, ind3]
        # result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        # reshape into (bs, n_characters, n_genera)
        mu_var = mu_var.reshape(-1, n_classes, self.n_characters * 2).permute(0, 2, 1)
        mu = mu_var[:, : self.n_characters, :]
        log_var = mu_var[:, self.n_characters :, :]

        # A = F.relu(A)

        # A = F.relu(A) + 1e-10
        # U = torch.zeros((bs, 144, 144)).cuda()
        # indices = torch.triu_indices(144, 144)
        # U[:, indices[0], indices[1]] = A  # set upper matrix
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

        return mu, log_var, indices

    def make_indices(self, maxpool_size, n_channels, out_length):
        arange = torch.arange(0, out_length * maxpool_size, maxpool_size)
        return torch.tile(arange, dims=[n_channels, 1])

    def decode(self, z: Tensor, indices) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        n_classes = z.shape[-1]

        X = self.dec1(z.permute(0, 2, 1).reshape(-1, self.n_characters, 1))
        X = self.dec_mp1(X)
        # self.dec_mp1(X, indices=torch.tile(self.dec_mp1_indices, [bs, 1, 1]))
        X = self.dec2(X)
        X = self.dec_mp2(X)
        # self.dec_mp2(X, indices=torch.tile(self.dec_mp2_indices, [bs, 1, 1]))
        X = self.dec3(X)
        X = self.dec_mp3(X[:, :, :277])
        # self.dec_mp3(
        # X[:, :, :277], indices=torch.tile(self.dec_mp3_indices, [bs, 1, 1])
        # )
        X = self.dec4(X)

        result = F.softmax(X, dim=1)
        result = result.reshape(-1, n_classes, 5, 836).permute(0, 2, 3, 1)
        return result

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
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

        # std =

        # eps = torch.randn_like(mu).view(-1, 6, 144)

        # multiply each of the matrices for each in batch, with vector of eps

        # samples = mu + torch.einsum("...jk,...ik", A, eps).reshape(
        #    -1, 144 * 6
        # )  # torch.einsum("...jk,...ik", A, eps)
        # samples = X + std * eps

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var, indices = self.encode(input)
        # _X = mu - mu.mean(dim=1).unsqueeze(1)
        # cov_q = torch.matmul(_X.transpose(2, 1), _X)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z, indices), input, mu, log_var]

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
        input = input.view(-1, 5, 836, n_classes)
        mu = args[2]
        log_var = args[3]
        labels = kwargs["labels"]
        losses = kwargs["losses"]
        labels = [x[0] for x in labels]
        class_indices = [self.ordered_genera.index(x) for x in labels]

        def my_matmul(A, B):
            return torch.einsum("...ik,...jk", A, B)

        loss = torch.tensor(0).cuda().float()
        logs = {}

        for weight, loss_name in losses:
            if loss_name == "kld":
                kld_loss = self.kld_loss_xo2(mu, labels)
                loss += weight * kld_loss
                logs["KLD"] = kld_loss.detach()
            elif loss_name == "hier_triplet":
                samples = self.reparameterize(mu, log_var)
                triplet_loss = self.triplet_loss(
                    samples, labels, class_indices, hierarchical=True
                )
                loss += weight * triplet_loss
                logs["Triplet"] = triplet_loss.detach()
            elif loss_name == "kld_indep_traits":
                kld_loss = self.kld_loss_indep_traits(mu, log_var, labels)
                loss += weight * kld_loss
                logs["KLD_indep_traits"] = kld_loss.detach()
            elif loss_name == "triplet":
                samples = self.reparameterize(mu, log_var)
                triplet_loss = self.triplet_loss(samples, labels, class_indices)
                loss += weight * triplet_loss
                logs["Triplet"] = triplet_loss.detach()
            elif loss_name == "recons":
                recons_loss = F.mse_loss(recons, input)
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


    def kld_loss_indep_traits(self, X, log_var, labels):
        n_chars = X.shape[1]
        n_species = X.shape[2]
        mu = X.mean(dim=[0,2])
        _X = X.mean(dim=0)
        cov_q = torch.matmul(_X, _X.T) / n_species
        kld_loss = 0.5 * (torch.matmul(mu, mu.T) + cov_q.trace() - n_chars - torch.log(torch.linalg.eigvalsh(cov_q)).sum())
        return kld_loss
    def kld_loss_indep_traits_old(self, X, log_var, labels):
        n_chars = X.shape[1]
        mu = X.permute(0,2,1).reshape(-1, n_chars)
        log_var = log_var.permute(0,2,1).reshape(-1, n_chars)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss
    def kld_loss_xo2(self, X, labels):
        n_characters = X.shape[1]
        n_classes = X.shape[-1]

        _X = X.mean(dim=0)
        _X = _X - _X.mean(dim=0, keepdim=True)
        #std_X = _X.std(dim=0, keepdim=True)
        #_X = _X / std_X
        cov_q = (
            torch.matmul(_X.T, _X) + torch.eye(n_classes).cuda() * 1e-4
        ) / n_characters
        mu_q = X.mean(dim=[0,2]).unsqueeze(1).repeat((1,n_classes)) # mean character should be 0... repeat it 144 times
        indices = [self.ordered_genera.index(x) for x in labels]
        T = self.T[indices][:, indices]
        inv_T = T.inverse()

        kld_loss = 0.5 * n_characters * torch.mean(
            torch.log(torch.linalg.eigvalsh(T)).sum()
            - torch.log(torch.linalg.eigvalsh(cov_q)).sum()
            - n_classes
            + self.trace(torch.matmul(torch.matmul(mu_q, inv_T), mu_q.T)) / n_characters
            + self.trace(torch.matmul(inv_T, cov_q)))
        return kld_loss


    def kld_loss_xo(self, X, labels):
        n_characters = X.shape[1]
        n_classes = X.shape[-1]

        _X = X.mean(dim=0)
        _X = _X - _X.mean(dim=1, keepdim=True)
        cov_q = (
            torch.matmul(_X.T, _X) + torch.eye(n_classes).cuda() * 1e-4
        ) / n_characters

        #_,cov_q = self.calc_V(X, labels)

        #cov_q = torch.diagonal(torch.einsum('ij,kl->ikjl',_X,_X), dim1=0,dim2=1).permute(2,0,1) + torch.eye(n_classes).cuda() * 1e-4

        mu_q = X.mean(dim=[0,2]).unsqueeze(1).repeat((1,n_classes)) # mean character should be 0... repeat it 144 times
        indices = [self.ordered_genera.index(x) for x in labels]
        T = self.V[indices][:, indices] + torch.eye(n_classes).cuda() * 1e-5
        inv_T = T.inverse()

        kld_loss = 0.5 * n_characters * torch.mean(
            torch.log(torch.linalg.eigvalsh(T)).sum()
            - torch.log(torch.linalg.eigvalsh(cov_q)).sum()
            - n_classes
            + self.trace(torch.matmul(torch.matmul(mu_q, inv_T), mu_q.T)) / n_characters
            + self.trace(torch.matmul(inv_T, cov_q)))
        return kld_loss
    def kld_loss(self, mu, labels):
        n_characters = mu.shape[1]
        n_classes = mu.shape[-1]
        _X = mu.mean(dim=0)  # average over the batch

        _X = _X - _X.mean(
            dim=0, keepdim=True
        )  # + torch.exp(0.5 * log_var).unsqueeze(1)
        cov_q = (
            torch.matmul(_X.T, _X) + torch.eye(n_classes).cuda() * 1e-2
        ) / n_characters

        mu_q = mu.mean(dim=[0,1])
        indices = [self.ordered_genera.index(x) for x in labels]
        T = self.T[indices][:, indices]
        inv_T = T.inverse()

        kld_loss = 0.5 * torch.mean(
            torch.log(torch.linalg.eigvalsh(T)).sum()
            - torch.log(torch.linalg.eigvalsh(cov_q)).sum(dim=0)
            - 144
            + torch.matmul(torch.matmul(mu_q, inv_T), mu_q.T)
            + self.trace(torch.matmul(inv_T, cov_q))
        )
        return kld_loss

    def triplet_loss(self, samples, labels, class_indices, hierarchical=False):
        sampled_triplets = self.batchminer(samples, labels, class_indices, hierarchical)
        triplet_loss = torch.stack(
            [
                F.triplet_margin_loss(
                    triplet[0],
                    triplet[1],
                    triplet[2],
                    margin=triplet[3],
                )
                for triplet in sampled_triplets
            ]
        ).mean()
        return triplet_loss

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
        tree = matrix_to_tree(mu.cpu().detach().numpy(), labels)
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
