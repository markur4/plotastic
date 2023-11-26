# %%


import os
from plotastic.utils.subcache import SubCache

# %%
### Define Home
home = os.path.join(
    os.path.expanduser("~"),
    ".cache",
)

# == Define SubCaches ==================================================
#' Define a Memory object for different purposes
MEMORY_UTILS = SubCache(
    location=home,
    assert_parent=".cache",
    subcache_dir="plotastic_utils",
)
# MEMORY_PLOTTING = SubCache(
#     location=home,
#     assert_parent=".cache",
#     subcache_dir="plotastic_plotting",
# )

### Cache like this:
# def sleep(seconds):
#     import time
#     time.sleep(seconds)

# sleep = caches.MEMORY_UTILS.subcache(sleep)

# == Utilities ====================================
if __name__ == "__main__":
    pass
    # %%
    ### View Contents
    MEMORY_UTILS.list_dirs()
    # %%
    # Clear Caches
    # MEMORY_UTILS.clear()
