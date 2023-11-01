#
# %% Imports


from typing import Optional, List

import os
import shutil

import icecream as ic

from joblib import Memory


# %% CACHE: Make cache dir


def cache_get_dir(package_name: str = "plotastic") -> str:
    """Returns a cache directory in the ".cache" directory within the systems home directory.

    :param package_name: Name of the package to be cached. It's the foldername within
        `~/.cache`, defaults to "plotastic"
    :type package_name: str, optional
    :return: _description_
    :rtype: str
    """
    home_directory = os.path.expanduser("~")
    package_name = "plotastic"
    ### Make a subfolder in .cache for this package
    cache_directory = os.path.join(home_directory, ".cache", package_name)
    # cache_directory = os.path.join(home_directory, f".{package_name}_cache")
    return cache_directory


def cache_init(package_name: str = "plotastic", verbose=False) -> Memory:
    """Initializes a cache directory in the home directory and returns a Memory object
    to cache functions like this: `ut.MEMORY.cache(func)` (It is discouraged to use it as a
    decorator)

    :param package_name: Name of the package to be cached. It's the foldername within
        `~/.cache`, defaults to "plotastic",
    :type package_name: str, optional

    :return: Memory object
    :rtype: Memory
    """
    cache_dir = cache_get_dir(package_name=package_name)

    if verbose:
        print(f"#! Cache directory: {cache_dir}")

    memory = Memory(cache_dir, verbose=0)
    return memory

### Initialize Memory object
# * We capitalize it to make it look like a constant, we intend to initialize it only once
MEMORY = cache_init()


# %% CACHE: List what's Cached


def cache_list(
    mem: Optional[Memory] = None,
    detailed: bool = False,
    max_depth: int = 3,
) -> List[str]:
    """
    Returns a list of cache directories.

    :param mem: An instance of Memory class. Default is None.
    :type mem: Optional[Memory], optional
    :param detailed: if True, returns all cache directories with full paths. Default is
        False.
    :type detailed: bool, optional
    :param max_depth: The maximum depth to search for cache directories. Default is 4.
    :type max_depth: int, optional
    :return: List[str], a list of cache directories.
    """
    cache_dir = cache_get_dir()

    mem = mem if not mem is None else MEMORY

    caches = []
    for root, dirs, _files in os.walk(cache_dir):
        # * Don't go too deep: 'joblib/plotastic/example_data/load_dataset/load_dataset',
        depth = root[len(cache_dir) :].count(os.sep)
        if not detailed and depth > max_depth:
            continue
        for dir in dirs:
            # * Don't need to check for 'joblib' because it's not a subdirectory of cache_dir
            # * Exclude subdirectories like "c1589ea5535064b588b2f6922e898473"
            if len(dir) >= 32 or dir == "joblib":
                continue
            # * Return every path completely
            if detailed:
                caches.append(os.path.join(root, dir))
            else:
                dir_path = os.path.join(root, dir)
                dir_path = dir_path.replace(cache_dir, "")
                if dir_path.startswith("/"):
                    dir_path = dir_path[1:]
                caches.append(dir_path)
    return caches


if __name__ == "__main__":
    cache_list()

# %% List what's Cached as objects


def cache_list_objects(mem: Memory = None):
    """Return the list of inputs and outputs from `mem` (joblib.Memory cache)."""
    mem = mem if not mem is None else MEMORY

    caches = []

    for item in mem.store_backend.get_items():
        path_to_item = os.path.split(
            os.path.relpath(item.path, start=mem.store_backend.location)
        )
        result = mem.store_backend.load_item(path_to_item)
        input_args = mem.store_backend.get_metadata(path_to_item).get("input_args")
        caches.append((input_args, result))
    return caches


# %% Clear Cache


def cache_clear(what: str | os.PathLike = "all") -> None:
    if what == "all":
        MEMORY.clear(warn=False)
    else:
        cache_dir = MEMORY.location
        subcache_dir = os.path.join(cache_dir, what)
        if os.path.exists(subcache_dir):
            shutil.rmtree(subcache_dir)
        else:
            ic(f"#! '{what}' not found in cache")


if __name__ == "__main__":
    pass
    cache_clear(what="joblib/plotastic")
    cache_list()
