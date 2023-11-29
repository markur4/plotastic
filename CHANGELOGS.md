
# 0.1.1
- Legends
  - Added `legend_kws` parameter to all multiplots
    - It seems strange to enforce `.edit_legend()` in chain
    - Also, the multiplot should decide, which legend should be
      displayed (e.g. by correct order of calling `.edit_legend()`
      inbetween or after `.fillaxes()`)
- Added Documentation notebooks to Readme
- Runtime config `plotting.rc`
  - `set_style()` now passes any known matplotlib style to
    `matplotlib.style.use()`
- Legend is now outside of the plot no matter the figure width!

# 0.1.0
- Initial release