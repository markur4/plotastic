"""Removes Parts from README.md that PyPi can't handle by removing parts
enclosed by a marker line"""

# %%
import argparse

# %%


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", default="README.md")



# %%
def open_readme(path: str) -> str:
    with open(path, "r") as f:
        README = f.read()
    return README


def write_readme(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


# %%
if __name__ == "__main__":
    # readme_in = os.path.join("..", "README.md") # ::
    # readme_out = os.path.join("..", "README_pypi.md") # ::
    args = parser.parse_args() # ::
    readme_in = args.input # ::
    readme_out = "README_pypi.md" # ::

    README = open_readme(readme_in)
    print(README)

    # %%
    split = README.split("<!-- REMOVE FOR PYPI -->")
    split
    # %%
    ### Remove those parts from readme that end with <!-- <<< REMOVE FROM PYPI  -->
    split_r = [p for p in split if not "<!-- REMOVESTART -->" in p]

    joined = "\n".join(split_r)

    # %%
    write_readme(readme_out, joined)
