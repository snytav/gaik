import glob
import os

def list_files_by_mask(directory, mask):
    """
    Returns a list of files in a directory that match a given mask.

    Args:
        directory: The directory to search in.
        mask: The file mask to use (e.g., "*.txt", "image*.png").

    Returns:
        A list of file paths that match the mask.
    """
    search_path = os.path.join(directory, mask)
    return glob.glob(search_path)

# Example usage:
directory_path = "/path/to/your/directory"  # Replace with the actual directory path
file_mask = "*.txt"
matching_files = list_files_by_mask(directory_path, file_mask)

if matching_files:
    print("Files matching the mask:")
    for file_path in matching_files:
        print(file_path)
else:
    print(f"No files found matching the mask '{file_mask}' in '{directory_path}'.")


def sort_by_substring(string_list,substr_num):
    return sorted(string_list, key=lambda x: x.split('_')[substr_num])

def merge_pairs(nm):
    from itertools import groupby
    import itertools
    li = nm
    tuple_list = list(map(tuple, li))
    output_list = [[idx] + list(itertools.chain.from_iterable(list(filter(None, i[1:])) for i in e)) for idx, e in
                   groupby(tuple_list, lambda x: x[0])]
    return output_list


def get_layer_sequence():
    directory_path = '.'
    file_mask = '*put*.txt'
    matching_files = list_files_by_mask(directory_path, file_mask)
    sorted_files = sort_by_substring(matching_files,3)
    nm = [[l.split('/')[1].split('_')[0],l] for l in sorted_files]
    # turn pairs like these ['fuse', 'input-unn00000'], ['fuse', 'input-uanalytic00000']
    # into ['fuse', 'input-unn00000', 'input-uanalytic00000']
    nm_merged_pairs = merge_pairs(nm)

    return nm_merged_pairs




if __name__ == '__main__':

    matching_files = list_files_by_mask('.', '*put*.txt')
    for f in matching_files:
        n = len(f.split('_'))
        if n != 4:
           print(f)
           os.remove(f)
    lseq = get_layer_sequence()


