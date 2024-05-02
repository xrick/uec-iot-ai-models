# Import the sys module to access system-specific information.
import os
import sys
import pathlib
def checkEndianess():
        # Check if the byte order of the platform is "little" (e.g., Intel, Alpha) and display a corresponding message.
    if sys.byteorder == "little":
        return 1;
        # print("Little-endian platform.")
    else:
        # If the byte order is not "little," assume it's "big" (e.g., Motorola, SPARC) and display a corresponding message.
        # print("Big-endian platform.")
        return 2;
    
    # Display another blank line for clarity.
    print();

def ChkDirAndCreate(dir_path,op):
    if not pathlib.Path(dir_path).is_dir():
        print(f"{dir_path} does not exist, create...");
        if op == 1:
            os.mkdir(dir_path);
        else:
            os.makedirs(dir_path;)
        return 1;
    else:
        print(f"{dir_path} has existed...")
        return 2;


    