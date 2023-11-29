


# def _get_legend_width(labels: list[str]) -> float:
#     """Calculates the width of the legend in inches, taking fontsize
#     into account"""

#     ### Add legend title, which is hue
#     labels = [DA.dims.hue] + labels  # TODO: replace with self

#     ### Split by new lines and flatten
#     labels = [label.split("\n") for label in labels]
#     labels = [item for sublist in labels for item in sublist]
#     # print(labels)

#     ### Get length of longest level (or title)
#     max_label_length = max([len(label) for label in labels])

#     ### Convert label length to inches
#     #' 1 inch = 72 points, one character = ~10 points
#     fontsize = _get_fontsize_legend()
#     character_per_inch = 72 / fontsize
#     if "Narrow" in DA.font_mpl:  # TODO: replace with self
#         character_per_inch = character_per_inch * 0.8

#     legend_width = max_label_length / character_per_inch

#     ### Add more for the markers
#     #' When the legend title (hue) is the largest, no space needed
#     if len(DA.dims.hue) != max_label_length:
#         # legend_width += 0.5 # TODO reactivate
#         print("added marker width")

#     return legend_width