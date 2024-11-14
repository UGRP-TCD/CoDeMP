from cli import guide


def main():
    # Version info.
    version = "1.0.0"

    option = guide(version)

    if option == '1':
        print("=> You selected option 1.\n")
        ##############################
        # Add codes
    else:
        print("=> You selected option 2.\n")
        ##############################
        # Add codes


if __name__ == '__main__':
    main()
