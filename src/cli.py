"""This module is for the command line interface (CLI) of the CoDeMP."""


def guide(version: str) -> int:
    print(f"""===============================================================================
[ CoDeMP ]

** Before you start, please make sure you have the following:
  1. Make sure you run `requirements.txt` to install the required packages.
  2. You should add images in the `data/using` folder.

** Please select an option:
--------------------------------------------------------------
|  1. Generate a description from a file (one by one).       |
|  2. Generate series of descriptions from a folder.         |
|  3. Exit the program.                                      |
--------------------------------------------------------------

** Note:
  - Developed by. DGIST-UGRP-2024-No.22
  - Contact via. https://github.com/MintCat98
  - Version: {version}

===============================================================================
""")

    # Check an input validation
    valid = ['1', '2', '3']

    while True:
        option: int = input("Input an option: ")

        if option in valid:
            return option
        else:
            print(
                f"=> Invalid option: {option}. Please select a valid option(1 / 2 / 3).\n")
