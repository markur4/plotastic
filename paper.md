---
title: '`plotastic`: Bridging Plotting and Statistics in Python'
tags:
  - Python
  - plotting
  - statistics
  - seaborn
  - pingouin
authors:
  - name: Martin Kuric
    affiliation: 1 # (Multiple affiliations must be quoted)
    # equal-contrib: true # (Multiple authors must be indicated this way)    
    corresponding: true # (This is how to denote the corresponding author)
  - name: Regina Ebert
    affiliation: 1
    orcid: 0000-0002-8192-869X
    # equal-contrib: false # (Multiple authors must be indicated this way)
    corresponding: false # (This is how to denote the corresponding author)
affiliations:
 - name: Department of Musculoskeletal Tissue Regeneration, University of Würzburg, Germany
   index: 1
date: 11.11.2023  
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

`plotastic` addresses the challenges of transitioning from exploratory
data analysis to hypothesis testing in Python's data science ecosystem.
Bridging the gap between `seaborn` and `pingouin`, this library offers a
unified environment for plotting and statistical analysis. It simplifies
the workflow with a user-friendly syntax and seamless integration with
familiar `seaborn` parameters (y, x, hue, row, col). Inspired by
`seaborn`'s consistency, `plotastic` utilizes a `DataAnalysis` object to
intelligently pass parameters to `pingouin` statistical functions. The
library systematically groups the data according to the needs of
statistical tests and plots, conducts visualisation, analyses and
supports extensive customization options. In essence, `plotastic`
streamlines the process, translating `seaborn` parameters into
statistical terms, providing researchers and data scientists with a
cohesive and user-friendly solution in Python.

# Statement of need

Python's data science ecosystem provides powerful tools for both
visualization and statistical testing. However, the transition from
exploratory data analysis to hypothesis testing can be cumbersome,
requiring users to switch between libraries and adapt to different
syntaxes.

`seaborn` has become a popular choice for plotting in Python, offering
an intuitive interface. Its statistical functionality is limited to
descriptive plots and bootstrapped confidence intervals
[@waskomSeabornStatisticalData2021]. The library `pingouin` offers an
extensive set of statistical tests, but it lacks integration with common
plotting capabilities [@vallatPingouinStatisticsPython2018].

`plotastic` addresses this gap by offering a unified environment for
plotting and statistical analysis. With an emphasis on user-friendly
syntax and integration with familiar `seaborn` parameters, it simplifies
the process for users already comfortable `seaborn`. The library ensures
a smooth workflow, from data import to hypothesis testing and
visualization.

# Example
The following code demonstrates how `plotastic` analyzes the
example dataset "fmri", similar to @waskomSeabornStatisticalData2021. 

```python
### IMPORT PLOTASTIC
import plotastic as plst

# IMPORT EXAMPLE DATA
DF, _dims = plst.load_dataset("fmri", verbose = False)

# EXPLICITLY DEFINE DIMENSIONS TO FACET BY
dims = dict(
    y = "signal",    # y-axis, dependent variable
    x = "timepoint", # x-axis, independent variable (within-subject factor)
    hue = "event",   # color,  independent variable (within-subject factor)
    col = "region"   # axes,   grouping variable
)

# INITIALIZE DATAANALYSIS OBJECT
DA = plst.DataAnalysis(
    data=DF,           # Dataframe, long format
    dims=dims,         # Dictionary with y, x, hue, col, row 
    subject="subject", # Datapoints are paired by subject (optional)
    verbose=False,     # Print out info about the Data (optional)
)

# STATISTICAL TESTS
DA.check_normality()   # Check Normality
DA.check_sphericity()  # Check Sphericity
DA.omnibus_rm_anova()  # Perform RM-ANOVA
DA.test_pairwise()     # Perform Posthoc Analysis

# PLOTTING
(DA
 .plot_box_strip()   # Pre-built plotting function initializes plot
 .annotate_pairwise( # Annotate results from DA.test_pairwise()
     include="__HUE" # Use only significant pairs across each hue
     ) 
)
```


![Example figure of `plotastic`. Image was generated using
version 0.1](How_To_Use/quick_example_fmri_files/example.png){ width=90%
label="fig:example"}


:Results from `DA.check_sphericity()`. `plotastic` assesses sphericity
after grouping the data by all grouping dimensions (hue, row, col). For
example, `DA.check_sphericity()` grouped the 'fmri' dataset by "region"
(col) and "event" (hue), performing four subsequent sphericity tests for
four datasets. []{label="tab:sphericity"} \label{tab:sphericity}

|    'region', 'event'                   | spher   |        W |     chi2 |   dof |   pval |   group count | n per group                              |
|:--------------------------------|:--------|------------:|------------:|------:|-------:|--------------:|:---------------------|
| 'frontal', 'cue'   | True    | 3.26e+20 | -462.7 |    44 |      1 |            10 | [14] |
| 'frontal', 'stim'  | True    | 2.45e+17 | -392.2  |    44 |      1 |            10 | [14] |
| 'parietal', 'cue'  | True    | 1.20e+20 | -452.9 |    44 |      1 |            10 | [14] |
| 'parietal', 'stim' | True    | 2.44e+13 | -301.9 |    44 |      1 |            10 | [14] |



:Results of `DA.omnibus_rm_anova()`. `plotastic` performs one two-factor
RM-ANOVA per axes (grouping the data by row and col dimensions) using x
and hue as the within-factors. For this example, `DA.omnibus_rm_anova()`
grouped the 'fmri' dataset by "region" (col), performing two subsequent
two-factor RM-ANOVAs. Within-factors are "timepoint" (x) and "event"
(hue). For conciceness, GG-Correction and effect sizes are not shown.
[]{label="tab:RMANOVA"} \label{tab:RMANOVA}

|     'region'          | Source                   |       SS | ddof1 | ddof2 |        MS |       F |       p-unc     | stars     |
|:----------------------|:-------------------------|---------:|------:|------:|----------:|--------:|----------------:|:-----------|------------:|
| 'parietal' | timepoint         | 1.583  |       9 |     117 | 0.175  | 26.20 | 3.40e-24 | ****    |
| 'parietal' | event             | 0.770 |       1 |      13 | 0.770  | 85.31 | 4.48e-07 | ****    |
| 'parietal' | timepoint * event | 0.623 |       9 |     117 | 0.069 | 29.54 | 3.26e-26 | ****     |
| 'frontal'  | timepoint         | 0.686 |       9 |     117 | 0.076 | 15.98 | 8.28e-17 | ****    |
| 'frontal'  | event             | 0.240 |       1 |      13 | 0.240  | 23.44  | 3.21e-4 | ***      |
| 'frontal'  | timepoint * event | 0.242 |       9 |     117 | 0.026 | 13.031 | 3.23e-14 | ****     |




# Overview


\begin{comment}
========================================================================
PROMPT:
Plese re-write the overview section completely using the following
information. 
Do not exceed 700 words. Be concise and
summarize points if they appear repeatedly, but give your best to not
omit important or interesting details. This section should stand on its
own and may reach technical complexity, but should be
understandable by anyone knowing the basics of matplotlib, `seaborn` and
pandas. NEVER use bullet points, use only full sentences that are
logically connected.

Use this information to write the overview section:

- input format of Data: 
  - Long-format pandas dataframe
  - similar to seaborn, uses tidy format [@wickhamTidyData2014a] for
    compatibility
  - 

- How to improve user friendliness
  - Users interface with `plotastic` with only one `DataAnalysis` Object
    that contains all tools for plotting and statistical analysis
  - Plots are the most intuitive way of understanding your data, and
    all subsequent statistical analysis can rely on the same parameters.
  - `seaborn` relies on the parameters hue, row, col to create
    'facetted' subplots, of which each plot y against x. This allows for
    rapid and intuitive exploration of multidimensional relationships. 
    `plotastic` extends this strength to statistical analysis 
  - `plotastic` stores these `seaborn` parameters during initialisation of
  the `DataAnalysis` Object 
    - automatically translates `seaborn` parameters into statistical terms 
    -  example code for example dataset from "fmri" (see # comments how `plotastic` translates seaborn
       parameters into statistical terms)
      ```python
        # dims is short for dimensions
        dims = dict(         # STATISTICAL TERM:
          y = "signal",     # y-axis, dependent variable
          x = "timepoint",  # x-axis, independent variable (within-subject factor)
          hue = "event",    # color,  independent variable (within-subject factor)
          col = "region",   # axes,   grouping variable
          row = "xxx"       # axes,   grouping variable
        )
        ```
  - This will minimize user input and increase user friendliness
    - The user only chooses between parametric and non-parametric tests
    - and for some cases, `plotastic` automatically chooses the paired/unpaired
      version of the test if a subject was specified during
      initialisation of the `DataAnalysis` Object.

- How `plotastic` integrates statistical analysis and plotting
  - How `plotastic` iterates through data: `plotastic` implements several iterators
    that group the data by different dimensions depending on the tests
    and plotting requirements. 
    - **Normality testing**: Data is grouped by all grouping dimensions and
      also the x-axis, yielding tables that each contain the datapoints of
      individual samples. 
    - **Sphericity and and Homoscedasticity testing**: Data is grouped by all
      grouping dimensions (hue, row, col), yielding Tables that contain
      a series of samples that's indexed by the x-axis. (\autoref{tab:sphericity})
    - **Omnibus and Posthoc analyses**: Data is grouped by by row and col dimensions in parallel to the `matplotlib` axes,
      yielding Tables that contain the complete data of one axes (both hue and x) and then performs one two-factor
      RM-ANOVA per axes  using x and hue as the within/between-factors. (\autoref{tab:RMANOVA})
  

- How plots are made and customized
  - pre-defined plotting functions initialize a `matplotlib` figure
    through `plt.subplots()` and return a `DataAnalysis` Object to
    enable chaining of methods
  - Axes are filled using axes level `seaborn` plotting functions (e.g.
    `sns.boxplot()`), utilizing the automated aggregation of datapoints
    into means and display of error bars.
  - keyword arguments are passed to these `seaborn` functions, allowing
    the same degree of customization as in seaborn.
  - The hidden state and figure aesthetics is accessed by either by
    chaining further `DataAnalysis` methods or by writing subsequent
    lines of common `matplotlib` code that overrides `plotastic`
    settings. Figures are exported by calling `plt.savefig()`.


- At version 0.1, `plotastic` supports the pingouin implementation of
  classical assumption and hypothesis testing, including their
  parametric/non-parametric and paired/non-paired variants.
  - Assumptions: Normality, Homoscedasticity, Sphericity
  - Omnibus Tests: RM-ANOVA, ANOVA, Friedman, Kruskal-Wallis
  - Posthoc tests: Implementation of `pingouin.pairwise_tests()` which
    includes (paired) t-tests, Wilcoxon and Mann-Whitney-U.




========================================================================
\end{comment}


The functionality of `plotastic` revolves around a seamless integration
of statistical analysis and plotting, leveraging the capabilities of
`pingouin`, `seaborn` and `matplotlib`
[@vallatPingouinStatisticsPython2018; @waskomSeabornStatisticalData2021;
@hunterMatplotlib2DGraphics2007]. It utilizes long-format `pandas`
`DataFrames` as its primary input, aligning with the conventions of
`seaborn` and ensuring compatibility with existing data structures
[@wickhamTidyData2014a; @mckinneyPandasFoundationalPython2011].

`plotastic` was inspired by `seaborn`'s intuitive and consistent usage
of the same set of parameters (y, x, hue, row, col) found in each of its
plotting functions [@waskomSeabornStatisticalData2021]. These parameters
intuitively delineate the data dimensions plotted, yielding 'facetted'
subplots, each presenting y against x. This allows for rapid and
insightful exploration of multidimensional relationships. `plotastic`
extends this principle to statistical analysis by storing these
 `seaborn` parameters (referred to as dimensions) in a `DataAnalysis`
object and intelligently passing them to statistical functions of the
`pingouin` library. This approach is based on the impression that most
decisions during statistical analysis can be derived from how the user
decides to arrange the data in a plot. This approach also prevents code
repetition and streamlines statistical analysis. For example, the
subject keyword is specified only once during `DataAnalysis`
initialisation, and `plotastic` selects the appropriate paired or
unpaired version of the test. Using `pingouin` alone requires the user to
manually pick the correct test and to repeatedly specify the subject
keyword in each testing function.

In essence, `plotastic` translates plotting parameters into their
statistical counterparts. This translation minimizes user input and also
ensures a coherent and logical connection between plotting and
statistical analysis. The goal is to allow the user to focus on choosing
the correct statistical test (e.g. parametric vs. non-parametric) and
worry less about specific implementations.

At its core, `plotastic` employs iterators to systematically group data
based on various dimensions, aligning the analysis with the distinct
requirements of tests and plots. Normality testing is performed on each
individual sample, which is achieved by splitting the data by all
grouping dimensions and also the x-axis (hue, row, col, x). Sphericity
and homoscedasticity testing is performed on a complete sampleset listed
on the x-axis, which is achieved by splitting the data by all grouping
dimensions (hue, row, col)  (\autoref{tab:sphericity}). For omnibus and
posthoc analyses, data is grouped by the row and col dimensions in
parallel to the `matplotlib` axes, before performing one two-factor
analysis per axes using x and hue as the within/between-factors.
(\autoref{tab:RMANOVA}).

`DataAnalysis` visualizes data through predefined plotting functions
designed for drawing multi-layered plots. A notable emphasis within
`plotastic` is placed on showcasing individual datapoints alongside
aggregated means or medians. In detail, each plotting function
initializes `matplotlib` figure and axes using `plt.subplots()` while
returning a `DataAnalysis` object for method chaining. Axes are
populated by `seaborn` plotting functions (e.g., `sns.boxplot()`),
leveraging automated aggregation and error bar displays. Keyword
arguments are passed to these `seaborn` functions, ensuring the same
degree of customization as in seaborn. Users can further customize plots
by chaining DataAnalysis methods or by applying common `matplotlib` code
to override plotastic settings. Figures are exported using
`plt.savefig()`.

For statistics, `plotastic` integrates with the `pingouin` library to
support classical assumption and hypothesis testing, covering
parametric/non-parametric and paired/non-paired variants. Assumptions
such as normality, homoscedasticity, and sphericity are tested. Omnibus
tests include two-factor RM-ANOVA, ANOVA, Friedman, and Kruskal-Wallis.
Posthoc tests are implemented through `pingouin.pairwise_tests()`,
offering (paired) t-tests, Wilcoxon, and Mann-Whitney-U.

To sum up, plotastic stands as a unified and user-friendly solution
catering to the needs of researchers and data scientists, seamlessly
integrating statistical analysis with the power of plotting in Python.
It streamlines the workflow, translates `seaborn` parameters into
statistical terms, and supports extensive customization options for both
analysis and visualization.






# Acknowledgments
This work was supported by the Deutsche Forschungsgemeinschaft (DFG) SPP
microBONE grants EB 447/10-1 (491715122), JA 504/17-1, HO 4462/1-1
(401358321), We thank the Elite Netzwerk Bayern and the Graduate School
of Life Sciences of the University of Würzburg.



# References