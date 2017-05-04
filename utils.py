import pickle

val_names_file = 'val.txt'
val_labels_file = 'ILSVRC2015_clsloc_validation_ground_truth.txt'
map_file = 'map_clsloc.txt'


# Return dictionary where key is validation image name and value is class label
# ILSVRC2012_val_00000001: 490
# ILSVRC2012_val_00000002: 361
# ILSVRC2012_val_00000003: 171
# ...
def get_val_ground_dict():
    # Table would be better? but keep dict
    d_labels = {}
    i = 1
    with open(val_labels_file) as f:
        for line in f:
            tok = line.split()
            d_labels[i] = int(tok[0])
            i += 1

    d = {}
    with open(val_names_file) as f:
        for line in f:
            tok = line.split()
            d[tok[0]] = d_labels[int(tok[1])]
    return d


# Get list of folders with order as in map_file
# Useful when we want to have the same splits (taking every n-th class)
def get_ordered_folders():
    folders = []

    with open(map_file) as f:
        for line in f:
            tok = line.split()
            folders.append(tok[0])
    return folders


# Returns dictionary where key is folder name and value is label num as int
# n02119789: 1
# n02100735: 2
# n02110185: 3
# ...
def get_label_dict():
    d = {}
    with open(map_file) as f:
        for line in f:
            tok = line.split()
            d[tok[0]] = int(tok[1])
    return d


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

