import os

def remove_empty_dirs(directory):
    """
    Recursively removes all empty directories within the specified directory.
    """
    for dirpath, dirnames, _ in os.walk(directory, topdown=False):
        for dirname in dirnames:
            dir_to_check = os.path.join(dirpath, dirname)
            if not os.listdir(dir_to_check):  # Check if directory is empty
                os.rmdir(dir_to_check)
                print(f"Removed empty directory: {dir_to_check}")

if __name__ == "__main__":
     # dir_path = input("Enter the directory path: ").strip()
    dir_path = "../save_model_data/all-combos-4-features-exact-gp"
    if os.path.isdir(dir_path):
        remove_empty_dirs(dir_path)
        print("Cleanup complete.")
    else:
        print("Invalid directory path.")
