# first line: 112
def make_DA_statistics(dataset: str = "qpcr") -> plst.DataAnalysis:
    """Makes a DA object with every possible data stored in it

    :param dataset: "tips", "fmri", or "qpcr"
    :type dataset: str
    """

    ### ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ### Example Data that's Paired, so we can use tests for paired data
        assert dataset not in ["tips"], f"{dataset} is not paired"

        ### Load example data
        DF, dims = plst.load_dataset(dataset, verbose=False)

        ### Init DA
        DA = plst.DataAnalysis(DF, dims, subject="subject", verbose=False)

        ### Assumptions
        DA.check_normality()
        DA.check_homoscedasticity()
        DA.check_sphericity()

        ### Omnibus
        DA.omnibus_anova()
        DA.omnibus_rm_anova()
        DA.omnibus_kruskal()
        DA.omnibus_friedman()

        ### Posthoc
        DA.test_pairwise()

    return DA
