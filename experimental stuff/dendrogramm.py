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

levelkeys_all = DA.levelkeys_all
levelkeys = DA.levelkeys
leveldict = DA.levels_dict_factor

DF = DF.set_index(DA.factors_all)

# %%


DA.levels_combocount()


# %%

Z = DA._link_levelkeys()
dendro = DA.levels_dendrogram()


# %% Calculate the inconsistency matrix

R = sch.inconsistent(Z)
R.shape

# %% Plot the inconsistency matrix
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap_inconsistency(matrix: np.array, **kwargs):
    ### Create a heatmap of the inconsistency matrix
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        cmap="viridis",
        # annot=True,
        fmt=".2f",
        cbar=True,
        **kwargs,
    )
    # plt.colorbar(fig.axes[0].collections[0])
    plt.title("Inconsistency Matrix")
    plt.xlabel("Depth (Levels in the Hierarchy)")
    plt.ylabel("Levelkeys [Cluster Index]")
    plt.show()


plot_heatmap_inconsistency(matrix=R)

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


#%% Traverse the tree
for leaf in tree.get_leaves():
    print("Leaf Node:", leaf.id, "Data Point:", leaf.pre_order(), "Cluster ID:", leaf.dist)


# %% Create a dendrogram



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
