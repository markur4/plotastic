"""Removes Parts from README.md that PyPi can't handle by removing parts
enclosed by a marker line"""

#%%
import os


def main():
    readme = os.path.join("..", "README.md")
    
    with open(readme, 'r') as f:
        README = f.read()  
        
    print(README)  
    
    README.split()
    


if __name__ == '__main__':
    main()