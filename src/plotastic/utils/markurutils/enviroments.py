from markurutils.builtin_types import printable_dict


def check_dependencies(deps: tuple | str, hard=True, extramessage: str = None):
    """Tries to import list of modules and prints out missining modules"""
    if isinstance(deps, str):
        deps = (deps,)

    missing_deps = {}
    for dependency in deps:
        try:
            __import__(dependency)
        except ImportError as e:
            # missing_deps.append(f"{dependency}: {e}")
            missing_deps[dependency] = str(e)
    if missing_deps:
        if hard:
            raise ImportError("Unable to import REQUIRED dependencies:\n"
                              + printable_dict(missing_deps, key_adjust=15, print_type=False))
        else:
            print("Unable to import OPTIONAL dependencies:\n"
                  + printable_dict(missing_deps, key_adjust=15, print_type=False))
            # + "\n".join(missing_deps))
    if extramessage:
        print(extramessage)
    return missing_deps









