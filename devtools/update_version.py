"""Helper script that updates the version number in pyproject.toml
"""


# %%
import argparse
import toml

# %%


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", default="pyproject.toml")
parser.add_argument("-o", "--output", default="pyproject.toml")




# %%
def open_toml(input:str) -> dict:
    with open(input, "r") as f:
        toml_dict = toml.load(f)
    return toml_dict

def write_toml(output:str, toml_dict:dict) -> None:
    with open(output, "w") as f:
        toml.dump(toml_dict, f)

if __name__ == "__main__":
    # input = "../pyproject.toml" # ::
    # output = "../pyproject.toml" # ::
    args = parser.parse_args() # ::
    input = args.input #::
    output = args.output #::
    
    tomldict = open_toml(input)
    print(tomldict)

    #%%
    ### get version of 
    version = tomldict["project"]["version"]
    version
    #%%
    ### Add +1 to the last digit of the version number
    version_split = version.split(".")
    version_split[-1] = str(int(version_split[-1]) + 1)
    new_version = ".".join(version_split)
    new_version
    
    #%%
    ### Update the version number in the dictionary
    tomldict["project"]["version"] = new_version
    
    #%%
    ### Write the new version number to the file
    write_toml(output, tomldict)
        