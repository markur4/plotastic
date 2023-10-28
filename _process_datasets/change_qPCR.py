#
# %%
import pandas as pd
import numpy as np

# %%
DF = pd.read_excel("qPCR_raw.xlsx")

# %%

DF = DF.drop(["date", "SS1", "Unnamed: 0"], axis=1)
DF
# %%
import random


def generate_random_number(num_digits):
    if num_digits <= 0:
        raise ValueError("Number of digits should be greater than 0")

    # Calculate the lower and upper bounds for the given number of digits
    lower_bound = 10 ** (num_digits - 1)
    upper_bound = (10**num_digits) - 1

    # Generate a random number within the specified bounds
    random_number = random.randint(lower_bound, upper_bound)

    return random_number


# Example: Generate a random 5-digit number
generate_random_number(5)


# %%
DF.rename(
    columns={"SS2": "Subject", "MSC Donor": "Donor"},
    inplace=True,
)
# %%
mapping_dict = {
    original_value: generate_random_number(4) for original_value in DF["Donor"].unique()
}

DF["Donor"] = DF["Donor"].replace(mapping_dict)
print(DF["Donor"].unique())

# %%
DF["Gene"] = DF["Gene"].astype("category")
DF["Class"] = DF["Class"].astype("category")
DF["Method"] = DF["Method"].astype("category")
DF["Donor"] = DF["Donor"].astype("category")
DF

# %%
s = []
for i, row in DF[["Gene", "Class"]].iterrows():
    # print(row)
    val = (row["Gene"], row["Class"])
    if val not in s:
        s.append(val)
s
# %%

print(DF)

print(DF["Class"].unique())
classes = [
    "ECM & Adhesion" "Signaling" "Bone Metabolism" "Chemokines" "Cytokines" "MMPs"
]



# %%
# print(DF["Gene"].cat.categories)

old_genes = [
    "BCL6",
    "BMP4",
    "BTG2",
    "CXCL12",
    "CXCL8",
    "DCN",
    "DKK1",
    "IL10RB",
    "IL24",
    "LOX",
    "MMP14",
    "MMP2",
    "Mucin1",
    "NOTCH2",
    "OPG",
    "PRICKLE1",
    "TGM2",
    "TNFRSF1A",
    "TRAF5",
]
# from chatGPT generate new genes
""" 
    ('Vimentin', 'ECM & Adhesion')
    ('FBN1', 'ECM & Adhesion')
    ('TNC', 'ECM & Adhesion')
    ('LOXL2', 'ECM & Adhesion')
    ('JAK2', 'Signaling')
    ('PTCH1', 'Bone Metabolism')
    ('SOST', 'Bone Metabolism')
    ('RUNX2', 'Bone Metabolism')
    ('STAT3', 'Signaling')
    ('CCL5', 'Chemokines')
    ('CCL20', 'Chemokines')
    ('IL2RG', 'Cytokines')
    ('IFNG', 'Cytokines')
    ('MMP9', 'MMPs')
    ('TIMP1', 'MMPs')
    ('WNT5A', 'Signaling')
    ('FZD4', 'Signaling')
    ('IL6R', 'Cytokines')
    ('TNFSF13', 'Cytokines')
"""
new_genes = [
    "Vimentin",
    "FBN1",
    "TNC",
    "LOXL2",
    "JAK2",
    "PTCH1",
    "SOST",
    "RUNX2",
    "STAT3",
    "CCL5",
    "CCL20",
    "IL2RG",
    "IFNG",
    "MMP9",
    "TIMP1",
    "WNT5A",
    "FZD4",
    "IL6R",
    "TNFSF13",
]

gene_dict = dict(zip(old_genes, new_genes))
gene_dict

DF["Gene"] = DF["Gene"].replace(gene_dict)
print(DF["Gene"].unique())
DF
# %%

# DF["Gene"] = DF["Gene"].cat.rename_categories(gene_dict)
# %%
DF["Gene"] = DF["Gene"].astype("category")
DF["Class"] = DF["Class"].astype("category")
DF["Method"] = DF["Method"].astype("category")
DF["Donor"] = DF["Donor"].astype("category")
DF

# %%
DF["Class"]: DF["Class"].cat.remove_unused_categories()
DF["Gene"]: DF["Gene"].cat.remove_unused_categories()

#%%
DF.columns
DF = DF[DF["Tcc"] == 24]
DF.drop("Tcc", axis=1, inplace=True)
DF

#%%
import seaborn as sns

sns.catplot(data=DF, y="FC", x="Gene", col="Class", kind="box", hue="Fraction", sharey=False, sharex=False, col_wrap=3)
# %%
DF.to_excel("qPCR.xlsx")

# %%
