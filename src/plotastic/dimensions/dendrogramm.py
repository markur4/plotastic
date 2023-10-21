#
# %% imports


import pandas as pd
import numpy as np

# from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200

import markurutils as ut
import plotastic as plst

# %% Load example data

DF, dims = plst.load_dataset("qpcr")
DA = plst.DataAnalysis(data=DF, dims=dims, verbose=True)
# ss = AN.data_get_samplesizes()
empty = DA.data_get_empty_groupkeys()

groupkeys = DA.levelkeys_all
leveldict = DA.levels_dict_factor

"""
each factor has multiple levels
these levels are combined with each other, I can represent that as a tree
A tree is too big, however and a pain to traverse
I could just calculate which factor has the biggest 'coverage' through their levels
Or maybe simply start with levelcounts..?
"""

#%%

from collections import defaultdict

def count_level_combinations(level_keys):
    # Create a dictionary to store the counts of level combinations
    level_combinations = defaultdict(int)

    # Iterate through the list of level keys
    for level_key in level_keys:
        # Iterate through all pairs of levels in the level key
        for i in range(len(level_key)):
            for j in range(i + 1, len(level_key)):
                # Sort the levels in alphabetical order to count combinations regardless of order
                combination = tuple(sorted((level_key[i], level_key[j])))
                level_combinations[combination] += 1

    return level_combinations

count_level_combinations(groupkeys)

# %%

def create_combination_dataframe(level_combinations):
    # Create a set to collect all unique levels
    unique_levels = set()
    
    # Collect all unique levels from the combinations
    for combination in level_combinations:
        unique_levels.update(combination)
    
    # Initialize a DataFrame with zeros and unique levels as columns and index
    df = pd.DataFrame(0, columns=list(unique_levels), index=list(unique_levels))
    
    # Update the DataFrame with the counts from level_combinations
    for combination, count in level_combinations.items():
        df.loc[combination[0], combination[1]] = count
        df.loc[combination[1], combination[0]] = count  # Ensure symmetry
    
    # Set the diagonal to 100%
    for level in unique_levels:
        df.loc[level, level] = 100
    
    return df

level_combination = count_level_combinations(groupkeys)

create_combination_dataframe(level_combination)

# %% count the number of unique levels per factor

levelcounts = {factor: len(levels) for factor, levels in leveldict.items()}
# * Class is a low number, but it is not represented everywhere

# %% For each factor, calculate the mean percentage how often each level occurs in the data


# """For each factor, calculate the mean percentage how often each level occurs in the data"""
# for factor, levels in leveldict.items():
#     # for level in levels:
#     # fmt: off
#     levels_coverage: pd.Series = (
#         DF
#         .index
#         .get_level_values(factor)
#         .value_counts(normalize=True)
#     )
    
#     # fmt: on
#     mean = round(levels_coverage.mean()*100, 0)
#     std = round(levels_coverage.std()*100, 1)
    
#     ### Multiply the mean by the number of levels
#     # * e.g. each level of Fraction occurs in 33% of the data. Because it is a factor
#     # * with 3 levels. Multiply 33% with 3 to get 100%, which proves that ech level of
#     # * Fraction is represented in the data

#     mult = mean * len(levels)
    
#     print(f"{factor.ljust(15)}: {mean}% ±{std}%\t {mult}%")
    
#     # ! this funciton is somewhat useless. 
    
# # %% Count for each level, how often it appears together with another level of all factors

# def counteach_level(level, levelkeys):
    
    
    
#     ### Initialize dictionary with levels as keys and counts as values
#     count_dict = {level: 0 for level in levelkeys}
#     for levelkey in levels:
#         if level in levelkey:
            

# for factors, levels in leveldict.items():
#     for level in levels:
        

# # %%

# # DF.set_index(["Fraction", "Class", "Gene", "Method"], inplace=True)
# DF.set_index(["Gene", "Fraction", "Class", "Method"], inplace=True)
# DF.index


# %% Calculate cluster hierarchy from dataframe index
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

### Example dataset (replace this with your MultiIndex)
# index = pd.MultiIndex.from_tuples(
#     [
#         ("IFNG", "F1", "ECM & Adhesion", "MACS"),
#         ("IFNG", "F2", "ECM & Adhesion", "MACS"),
#         ("TNFSF13", "F2", "Cytokines", "Wash"),
#         # Add more entries from your MultiIndex
#     ],
#     names=["Gene", "Fraction", "Class", "Method"],
# )


### Calculate the Jaccard similarity between combinations
def _jaccard_similarity(
    combination1: list | tuple, combination2: list | tuple
) -> float:
    """Calculates the Jaccard similarity:
    - We take two combinations:
    - e.g. ("A", 2, "C", "D")
    - and  ("A", 1, "C", "D"),
    - Measure the length of the intersection (=1)
    - Measure the length of the union (=7)
    - Divide the intersection by the union (1/7 = 0.14)

    :param combination1: _description_
    :type combination1: list | tuple
    :param combination2: _description_
    :type combination2: list | tuple
    :return: Similarity between two lists
    :rtype: float
    """
    ### Take unique values,
    # * Yes the Score might change, but relations between scores won't change
    set1 = set(combination1)
    set2 = set(combination2)

    ### Calculate Jaccard similarity
    len_intersection = len(set1.intersection(set2))
    len_union = len(set1) + len(set2) - len_intersection
    return len_intersection / len_union


print(_jaccard_similarity(("A", 2, "C", "D"), ("A", 2, "C", "D")))
print(_jaccard_similarity(("A", 2, "C", "D"), ("A", 1, "C", "D")))
print(_jaccard_similarity(("A", 2, "C", "D"), ("A", 1, "C", "DDD")))
print(_jaccard_similarity(("A", 2, "C", "D"), (9, 4, 3, 4)))

# %% Get Cluster hierarchy


def _find_cluster_in_levelkeys(method="ward") -> np.array:
    """_summary_
    :method: How to link distance matrix by sch.linkage.
        ["ward","single","complete","average"], defaults to "ward"
    :return: _description_
    :rtype: _type_
    """
    ### Take Unique values from the index
    # * The number of occurences of each index is not important
    # * They often represent technical replicates
    # * It's more important how often the levels occur together within one element
    ### levelkeys are the same as index after setting DF.index to all factors
    # levelkeys = DF.index
    levelkeys = DF.index.unique()
    # levelkeys = self.levelkeys_all

    ### Create a square distance matrix based on Jaccard similarity
    n = len(levelkeys)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = 1 - _jaccard_similarity(levelkeys[i], levelkeys[j])

    ### Perform hierarchical clustering
    Z = sch.linkage(dist_matrix, method="ward")

    return Z


Z = _find_cluster_in_levelkeys()
print(Z.shape)
print(DF.index.unique().shape)

# %% Use inconsistency matrix to help find clusters

R = sch.inconsistent(Z)
R.shape
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import seaborn as sns


### Create a heatmap of the inconsistency matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    R,
    cmap="viridis",
    # annot=True,
    fmt=".2f",
    cbar=True,
)

plt.title("Inconsistency Matrix")
plt.xlabel("Depth (Levels in the Hierarchy)")
plt.ylabel("Levelkeys [Cluster Index]")
plt.show()


"""
so in that heatmap I got a hierarchy depth of 4. I guess this number was determined
by sch.linkage? 


I see that the inconsistency takes values between 0-8. But it seems that even a depth of
2 is sufficient, because the inconsistency of the third depth level is only 3. So When I
decide to use only 2 clusters, my overall inconsistency or 'error' of that clustering will be smaller than 3?
"""

# %% Determine at which depth the inconsistency is below 30% of the maximum


def _calc_best_clusternumber(Z: np.array, threshold: float = 0.1) -> int:
    """Determines the best number of clusters based on the inconsistency matrix.

    :param Z: Linkage matrix from scipy.cluster.hierarchy.linkage()
    :type Z: np.array
    :param threshold: percentage of the maximum inconsistency we want to tolerate,
        defaults to .3
    :type threshold: float
    :return: Number of clusters sacrificing at most a percentage of the maximum
        inconsistency
    :rtype: int
    """

    ### Calculate inconsistencxy
    R = sch.inconsistent(Z)

    ### Calculate the maximum inconsistency
    max_inconsistency = threshold * np.max(R)

    ### Transpose the inconsistency matrix
    # * So that the first iteration level goes through hierarchy depth
    # R_t= R.T # ! not needed, just use correct slicing

    ### Calculate the depth at which the inconsistency is below a percentage of the maximum
    # * return the first found index where the max value is below a percentage of the maximum of
    # * the whole array. That depth is the best number of clusters where inconsistency
    # * is guaranteed to be below a percentage of the maximum inconsistency
    depth_return = 0
    for depth in range(R.shape[1]):
        if np.max(R[:, depth]) < max_inconsistency:
            depth_return = depth
            break

    if depth_return == 0:
        return R.shape[1]
    else:
        return depth_return


best = _calc_best_clusternumber(Z)

# %% Retrieve what levelkeys are int he same clusters.


def _get_clusterdict(Z, consistency_threshold: float = 0.5):
    """_summary_

    :param Z: _description_
    :type Z: _type_
    :param consistency_threshold: Max clusters is the original depth of Z, you can
        further reduce depth/numbers of clusters by defining a threshold percentage of maximum
        inconsistency you're willing to tolerate to reduce clusters, defaults to .1
    :type consistency_threshold: float, optional
    :return: _description_
    :rtype: _type_
    """

    ### retrieve cluster by cutoff height
    # cutoff_height = 0.5
    # clusters:np.array = sch.fcluster(Z, t=cutoff_height, criterion="distance")

    ### Make an Array that assigns a cluster-id to each distance
    # * At max we want 4 clusters, since we only have 4 dimensions for plotting
    # * (x,hue,row,col)
    # *
    maxclust = _calc_best_clusternumber(Z, threshold=consistency_threshold)
    # maxclust = len(self.factors_all)
    cluster_ids: np.array = sch.fcluster(
        Z,
        t=maxclust,
        criterion="maxclust",
        # criterion="maxclust_monocrit", # ! That returns a flat cluster
    )

    ### Cluster_ids are indexed by the order of the data points or levelkeys
    assert len(DF.index.unique()) == len(
        cluster_ids
    ), "Number of levels and cluster-assignments must match"

    # ==
    # == Iterate through levelkeys and cluster_ids and assign each levelkey to a cluster ==

    ### Initialize a dictionary for cluster_id as keys and list of levelkeys as values
    clusterdict = {id: [] for id in cluster_ids}
    for levelkey, cluster_id in zip(DF.index.unique(), cluster_ids):
        clusterdict[cluster_id].append(levelkey)

    return clusterdict


clusterdict = _get_clusterdict(Z)
clusterdict


# %% evaluate clusterdict

for cluster_id, levelkeys in clusterdict.items():
    pass

# %% test to_tree function

tree = sch.to_tree(Z)
dir(tree)
dict(tree.__dict__)
tree.get_count()


# %% Traverse the tree
# for leaf in tree.get_leaves():
#     print("Leaf Node:", leaf.id, "Data Point:", leaf.pre_order(), "Cluster ID:", leaf.dist)


# %% Create a dendrogram


def factors_dendrogram(clustering_method="ward", **dendrogramm_kws) -> dict:
    """Plots a dendrogramm that shows the hierarchical clustering of each levelkeys.
    It helps determining how the data is organized by each factor's levels.

    :param clustering_method: How to link distance matrix by sch.linkage.
        ["ward","single","complete","average"], defaults to "ward"
    :type clustering_method: str, optional
    :return: A dictionary with dendrogramm info and a matplotlib plot
    """

    Z = _find_cluster_in_levelkeys(method=clustering_method)

    # Create a dendrogram
    plt.figure(figsize=(3, len(Z) / 10))
    dendro: dict = sch.dendrogram(
        Z,
        labels=DF.index.unique(),
        # labels=self.levelkeys_all, #* Same as DF.index.unique() if factors are set as index
        orientation="right",
        **dendrogramm_kws,
    )

    ### annotate dendrogram
    # level_colors = {
    #     'Gene': 'b',
    #     'Fraction': 'g',
    #     'Class': 'r',
    #     'Method': 'c',
    #     # Add more levels and colors as needed
    # }
    # for leaf, d, level in zip(dendro['leaves'], dendro['dcoord'], branch_labels):
    #     color = level_colors.get(level, 'k')  # Default to black if level not found
    #     branch_x = np.mean(d[1:3])
    #     plt.text(branch_x, leaf, f'({level})', ha='center', va='center', color=color)

    plt.legend()

    plt.title("Combination Hierarchy")
    plt.xlabel("Linkage Distance")

    plt.show()

    return dendro


dendro = factors_dendrogram()
type(dendro)

# %% make the dendrogram simpler
### Not what I'm looking for
# dendro = factors_dendrogram(
#     # truncate_mode="lastp",
#     # show_contracted=True,
#     # p=12,
# )

# %% automate dendrogram coloring


import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt

# Example hierarchy tree structure (replace with your data)
data = np.random.random((10, 10))
Z = sch.linkage(data, method="ward")

# Example data - dictionary mapping levels (factors) to cluster colors
level_colors = {
    "Gene": "b",
    "Fraction": "g",
    "Class": "r",
    "Method": "c",
    # Add more levels and colors as needed
}

# Create a list of labels for the branches (customize this list)
branch_labels = list(level_colors.keys())

# Create the dendrogram with labels
dendro = sch.dendrogram(
    Z, labels=branch_labels, orientation="right", above_threshold_color="k"
)

# Annotate each branch with its level and color
for leaf, d, level in zip(dendro["leaves"], dendro["dcoord"], branch_labels):
    color = level_colors.get(level, "k")  # Default to black if level not found
    branch_x = np.mean(d[1:3])
    plt.text(branch_x, leaf, f"({level})", ha="center", va="center", color=color)

# Create a custom color legend based on the level_colors dictionary
legend_labels = list(level_colors.keys())
legend_handles = [
    plt.Line2D([0], [0], color=color, label=label)
    for label, color in level_colors.items()
]

# Display the legend
plt.legend(handles=legend_handles)


### Show the plot
plt.show()


# %% experiment with dendro object

print(dendro.keys())

# Annotate each branch with its level
for leaf, d in zip(dendro["leaves"], dendro["dcoord"]):
    # print(leaf, d)
    # level = branch_labels[leaf]
    branch_x = np.mean(d[1:3])
    # plt.text(branch_x, leaf, f'({level})', ha='center', va='center')


# %% print out hierarchy

from collections import OrderedDict

index = DF.index.unique()  # * same as DA.levelkeys_all

# Assign data points to clusters based on a cutoff height
cutoff_height = 0.5  # Adjust this threshold based on your needs
cluster_ids = fcluster(Z, t=cutoff_height, criterion="distance")

# Create an ordered dictionary to represent the hierarchy
cluster_hierarchy = OrderedDict()

# Iterate through the clusters and create the hierarchy
for i, cluster_id in enumerate(cluster_ids):
    levels = index[i]
    hierarchy = cluster_hierarchy
    for level in levels:
        hierarchy = hierarchy.setdefault(level, OrderedDict())
    hierarchy["Cluster"] = cluster_id


# Define a function to recursively print the hierarchy
def print_hierarchy(hierarchy, rank_position=0):
    for key, value in hierarchy.items():
        if key == "Cluster":
            print(f"Rank Position {rank_position}: Cluster {value}")
        else:
            print(f"Rank Position {rank_position}: {key}")
            print_hierarchy(value, rank_position + 1)


# Print the hierarchy
print_hierarchy(cluster_hierarchy)

# %% Anytree

# from anytree import Node, RenderTree

# # Create the root node
# root = Node("Root")

# ### Example cluster_hierarchy (customize this based on your data)
# # cluster_hierarchy = {
# #     'Gene': 'Cluster 1',
# #     'Fraction': 'Cluster 2',
# #     'Class': 'Cluster 3',
# #     'Method': 'Cluster 4',
# #     # Add more nodes and hierarchy as needed
# # }

# # Create nodes for each level and cluster
# for level, cluster in cluster_hierarchy.items():
#     level_node = Node(level, parent=root)
#     cluster_node = Node(cluster, parent=level_node)

# # Print the tree structure
# for pre, fill, node in RenderTree(root):
#     print(f"{pre}{node.name}")


# %%

# leveldict

# counts = dict()
# for key, value in cluster_hierarchy.items():
#     if key != "Cluster":
#         continue
#     else:

#         print(key, value)

# %%

from collections import defaultdict


def summarize_hierarchy(hierarchy, level_dict, rank_position=0, summary=None):
    if summary is None:
        summary = {factor: defaultdict(int) for factor in level_dict.keys()}

    for key, value in hierarchy.items():
        if key != "Cluster":
            factor = find_factor(key, level_dict)
            summary[factor][rank_position] += 1
            summarize_hierarchy(value, level_dict, rank_position + 1, summary)

    return summary


def find_factor(level, level_dict):
    for factor, levels in level_dict.items():
        if level in levels:
            return factor
    return None  # Handle cases where level is not found in level_dict


level_dict = leveldict
summary = summarize_hierarchy(cluster_hierarchy, level_dict)

# Print the summary
for factor, ranks in summary.items():
    print(f"Factor: {factor}")
    for rank, count in ranks.items():
        print(f"  Rank Position: {rank}, Count: {count}")


# %%
hä

# %%
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt

# Example hierarchy tree structure (replace with your data)
# data = np.random.random((10, 10))
# Z = sch.linkage(data, method='ward')

# Determine the number of leaves (number of branches)
num_leaves = len(DF.index.unique())

# Create a list of labels for the branches (customize this list)
branch_labels = ["Gene", "Fraction", "Class", "Method"]

# Ensure the number of labels matches the number of leaves
if num_leaves != len(branch_labels):
    raise ValueError(
        "Number of labels should match the number of leaves in the dendrogram"
    )

# Initialize dictionaries to store branch-level associations
branch_to_level = {}
level_to_branch = {}


def collect_branch_information(tree, current_branch):
    if "leaves" in tree:
        # This is a leaf node, assign the branch to the levels
        for leaf, label in zip(tree["leaves"], branch_labels):
            level_to_branch[leaf] = label
        return

    if "icoord" in tree and "ivl" in tree:
        # This is a non-leaf node, continue collecting branch information
        for i, label in zip(tree["icoord"], tree["ivl"]):
            branch_to_level[i] = label

    # Recursively traverse the tree for left and right branches
    if "icoord" in tree and len(tree["icoord"]) >= 4:
        collect_branch_information(
            {"icoord": tree["icoord"][:2], "ivl": tree["ivl"][:2]}, current_branch
        )
        collect_branch_information(
            {"icoord": tree["icoord"][2:], "ivl": tree["ivl"][2:]}, current_branch
        )


# Start collecting branch information
collect_branch_information(
    sch.dendrogram(Z, orientation="right"), current_branch=branch_labels[0]
)

# Create the dendrogram with labels
dendrogram = sch.dendrogram(Z, orientation="right")

# Annotate each branch with level information
for leaf, d in zip(dendrogram["leaves"], dendrogram["dcoord"]):
    level = level_to_branch[leaf]
    branch_x = np.mean(d[1:3])
    plt.text(branch_x, leaf, f"({level})", ha="center", va="center")

# Show the plot
plt.show()


# %%
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt

# Example hierarchy tree structure (replace with your data)
# Z = sch.linkage(np.random.random((10, 10)), method='ward')

# Create a list of labels for the branches (customize this list)
branch_labels = ["Gene", "Fraction", "Class", "Method"]

# Initialize dictionaries to store branch-level associations
branch_to_level = {}
level_to_branch = {}


def collect_branch_information(tree, current_branch):
    if "leaves" in tree:
        # This is a leaf node, assign the branch to the levels
        for leaf, label in zip(tree["leaves"], branch_labels):
            level_to_branch[leaf] = label
        return

    if "icoord" in tree and "ivl" in tree:
        # This is a non-leaf node, continue collecting branch information
        for i, label in zip(tree["icoord"], tree["ivl"]):
            branch_to_level[i] = label

    # Recursively traverse the tree for left and right branches
    if "icoord" in tree and len(tree["icoord"]) >= 4:
        collect_branch_information(
            {"icoord": tree["icoord"][:2], "ivl": tree["ivl"][:2]}, current_branch
        )
        collect_branch_information(
            {"icoord": tree["icoord"][2:], "ivl": tree["ivl"][2:]}, current_branch
        )


# Start collecting branch information
collect_branch_information(
    sch.dendrogram(Z, orientation="right"), current_branch=branch_labels[0]
)

# Create the dendrogram with labels
dendrog = sch.dendrogram(Z, labels=branch_labels, orientation="right")

# Annotate each branch with level information
for leaf, d in zip(dendrog["leaves"], dendrog["dcoord"]):
    level = level_to_branch[leaf]
    branch_x = np.mean(d[1:3])
    plt.text(branch_x, leaf, f"({level})", ha="center", va="center")

# Show the plot
plt.show()
