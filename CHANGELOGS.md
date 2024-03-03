# 0.1.2
### Available on:
- github

### New Features:
- None yet



# 0.1.1
### Available on:
- github
- PyPi

### New Features:
- Runtime config `plotting.rc`
  - `set_style()` now passes all available matplotlib styles to `matplotlib.style.use()`

### Experimental Features:
- Plot Paired Data by Joining subjects with line for each facet/x/hue-level! 
  - To my knowledge, the solutions provided by matplolib or seaborn are
    way too difficult. 
  - I implemented a solution that worked, but since I
    found a case where it didn't, this feature is experimental

    
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

