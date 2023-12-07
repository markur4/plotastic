
# 0.1.1
### Available on:
- github

### New Features:
- Plot Paired Data by Joining subjects with line for each facet/x/hue-level! 
  - To my knowledge, the solutions provided by matplolib or seaborn are
    way too difficult
    plot 
- Runtime config `plotting.rc`
  - `set_style()` now passes any known matplotlib style to
    `matplotlib.style.use()`
    
### Changes:
- Legends
  - Added `legend_kws` parameter to all multiplots
    - It seems strange to enforce `.edit_legend()` in chain
    - Also, the multiplot should decide, which legend should be
      displayed (e.g. by correct order of calling `.edit_legend()`
      inbetween or after `.fillaxes()`)

### Fixes:
- Rewrote .edit_titles_with_func() becasue it didn't work
- Legend is now outside of the plot no matter the figure width!

### Others:
- Added Documentation notebooks to Readme


# 0.1.0 - Initial Release
### Available on:
- github
- pypi