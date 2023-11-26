#
# %% Imports


from typing import Callable, List

import os
from pathlib import Path

from icecream import ic

from joblib import Memory

# from plotastic.utils import utils as ut


class SubCache(Memory):
    """Expands the joblib.Memory class with some useful methods.
    -
    - List directories within cache
    - List objects within cache
    - Adds subcache attribute, with benefits:
        - Subcache replaces module name in cache directory
        - More control over cache directories
        - Persistent caching, since IPythond passes a new location to
          joblib each time the Memory object is initialized
    - Doesn't work right if two SubCache Objects point cache the same function
    """

    def __init__(
        self, subcache_dir: str, assert_parent: str = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        ### Subfolder of location, overrides default subfolder by joblib
        self.subcache_dir = subcache_dir

        ### self.location/joblib/subcache
        self.subcache_path = os.path.join(
            self.location, "joblib", self.subcache_dir
        )

        ### Prevent joblib folders being created by wrong Interactive Windows
        if not assert_parent is None:
            parent_full = Path(self.location).absolute()
            parent = os.path.split(parent_full)[-1]
            assert (
                parent == assert_parent
            ), f"When Initializing joblib.Memory, we expected cache to be in {assert_parent}, but we ended up in {parent_full}"

    def list_dirs(
        self, detailed: bool = False, max_depth: int = 3
    ) -> List[str]:
        """
        Returns a list of cache directories.

        :param detailed: if True, returns all cache directories with
            full paths. Default is False.
        :type detailed: bool, optional
        :param max_depth: The maximum depth to search for cache
            directories. Default is 4.
        :type max_depth: int, optional
        :return: List[str], a list of cache directories.
        """

        subcache = self.subcache_path

        location_subdirs = []

        ### Recursive walking
        for root, dirs, _files in os.walk(subcache):
            #' Don't go too deep: 'joblib/plotastic/example_data/load_dataset/load_dataset',
            depth = root[len(subcache) :].count(os.sep)
            if not detailed and depth > max_depth:
                continue
            for dir in dirs:
                #' Don't need to check for 'joblib' because it's not a subdirectory of cache_dir
                #' Exclude subdirectories like "c1589ea5535064b588b2f6922e898473"
                if len(dir) >= 32 or dir == "joblib":
                    continue
                #' Return every path completely
                if detailed:
                    location_subdirs.append(os.path.join(root, dir))
                else:
                    dir_path = os.path.join(root, dir)
                    dir_path = dir_path.replace(subcache, "")
                    if dir_path.startswith("/"):
                        dir_path = dir_path[1:]
                    location_subdirs.append(dir_path)
        return location_subdirs

    def list_objects(self):
        """Return the list of inputs and outputs from `mem` (joblib.Memory
        cache)."""

        objects = []

        for item in self.store_backend.get_items():
            path_to_item = os.path.split(
                os.path.relpath(item.path, start=self.store_backend.location)
            )
            result = self.store_backend.load_item(path_to_item)
            input_args = self.store_backend.get_metadata(path_to_item).get(
                "input_args"
            )
            objects.append((input_args, result))
        return objects

    def subcache(self, f: Callable, **mem_kwargs) -> Callable:
        """Cache it in a persistent manner, since Ipython passes a new
        location to joblib each time the Memory object is initialized
        """
        f.__module__ = self.subcache_dir
        f.__qualname__ = f.__name__

        return self.cache(f, **mem_kwargs)


if __name__ == "__main__":
    home = os.path.join(
        os.path.expanduser("~"),
        ".cache",
    )

    def sleep(seconds):
        import time

        time.sleep(seconds)

    MEM = SubCache(location=home, subcache_dir="plotastic", verbose=True)

    sleep = MEM.subcache(sleep)
    # %%
    ### First time slow, next time fast
    sleep(1.4)
    # %%
    MEM.list_dirs()
    # %%
    MEM.clear()

    # %%
    ### Using different cache allows clearance of only that cache
    MEM2 = SubCache(location=home, subcache_dir="plotic2", verbose=True)

    def slep(seconds):
        import time

        time.sleep(seconds)

    sleep_cached2 = MEM2.subcache(slep)
    sleep_cached2(1.4)
    # %%
    MEM2.list_dirs()
    # %%
    MEM2.clear()
