from cli import guide
from program import run_by_option


def main():
    # Version info.
    version = "1.0.0"

    # CLI guide & Show options
    option = guide(version)
    # Option 3: Exit the program
    if option == 3:
        print("=> You selected option 3. Exit the program.")
        return

    # Run the program by the selected option
    run_by_option(option)


if __name__ == '__main__':
    main()
