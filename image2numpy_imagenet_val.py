# http://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10/35034287

from argparse import ArgumentParser
import numpy as np
import os
from scipy import misc
from utils import *


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-i', '--in_dir', help="Input directory with source images")
    parser.add_argument('-o', '--out_dir', help="Output directory for pickle files")
    args = parser.parse_args()

    return args.in_dir, args.out_dir


def process_folder(in_dir, out_dir):
    label_dict = get_label_dict()
    folders = get_ordered_folders()
    val_ground_dict = get_val_ground_dict()

    # Subsampling folders could be useful when we want to create smaller dataset
    # For example we want to use only every 10th class or first 100 classes (Below)

    # Here subsample folders
    # folders = folders[0::10]
    # folders = folders[:100]

    # Table contains labels that are associated with those folders
    labels_searched = []
    for folder in folders:
        labels_searched.append(label_dict[folder])

    print("Processing folder %s" % in_dir)
    labels_list = []
    images = []
    for image_name in os.listdir(in_dir):
        # Get label for that image
        # If it was resized using 'image_resizer_imagenet.py' script then we know that it has extension '.png'
        label = val_ground_dict[image_name[:-4]]

        # Ignore if it's not one of the subsampled classes
        if label not in labels_searched:
            continue
        try:
            img = misc.imread(os.path.join(in_dir, image_name))
            r = img[:, :, 0].flatten()
            g = img[:, :, 1].flatten()
            b = img[:, :, 2].flatten()

        except:
            print('Cant process image %s' % image_name)
            with open("log_img2np_val.txt", "a") as f:
                f.write("Couldn't read: %s" % os.path.join(in_dir, image_name))
            continue
        arr = np.array(list(r) + list(g) + list(b), dtype=np.uint8)
        images.append(arr)
        labels_list.append(label)

    data_val = np.row_stack(images)

    # Can add some kind of data splitting
    d_val = {
        'data': data_val,
        'labels': labels_list
    }
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pickle.dump(d_val, open(os.path.join(out_dir, 'val_data'), 'wb'))

    y_test = d_val['labels']
    count = np.zeros([1000])

    for i in y_test:
        count[i-1] += 1

    for i in range(1000):
        print('%d : %d' % (i, count[i]))

    print('SUM: %d' % len(y_test))

if __name__ == '__main__':
    in_dir, out_dir = parse_arguments()

    print("Start program ...")
    process_folder(in_dir=in_dir, out_dir=out_dir)
    print("Finished.")
