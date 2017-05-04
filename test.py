from argparse import ArgumentParser
import numpy as np
from PIL import Image
from utils import *
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# Test script

# Change this one to check other file


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-i', '--in_file', help="Input File with images")
    parser.add_argument('-g', '--gen_images', help='If true then generate big (10000 small images on one image) images',
                        action='store_true')
    parser.add_argument('-s', '--sorted_histogram', help='If true then histogram with number of images for '
                                                         'class will be sorted', action='store_true')
    args = parser.parse_args()

    return args.in_file, args.gen_images, args.sorted_histogram


def load_data(input_file):

    d = unpickle(input_file)
    x = d['data']
    y = d['labels']

    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))

    return x, y

if __name__ == '__main__':
    input_file, gen_images, hist_sorted  = parse_arguments()
    x, y = load_data(input_file)

    # Lets save all images from this file
    # Each image will be 3600x3600 pixels (10 000) images

    blank_image = None
    curr_index = 0
    image_index = 0

    print('First image in dataset:')
    print(x[curr_index])

    if not os.path.exists('res'):
        os.makedirs('res')

    if gen_images:
        for i in range(x.shape[0]):
            if curr_index % 10000 == 0:
                if blank_image is not None:
                    print('Saving 10 000 images, current index: %d' % curr_index)
                    blank_image.save('res/Image_%d.png' % image_index)
                    image_index += 1
                blank_image = Image.new('RGB', (36*100, 36*100))
            x_pos = (curr_index % 10000) % 100 * 36
            y_pos = (curr_index % 10000) // 100 * 36

            blank_image.paste(Image.fromarray(x[curr_index]), (x_pos + 2, y_pos + 2))
            curr_index += 1

        blank_image.save('res/Image_%d.png' % image_index)

    graph = [0] * 1000

    for i in range(x.shape[0]):
        # Labels start from 1 so we have to subtract 1
        graph[y[i]-1] += 1

    if hist_sorted:
        graph.sort()
        
    x = [i for i in range(1000)]
    ax = plt.axes()
    plt.bar(left=x, height=graph, color='darkblue', edgecolor='darkblue')
    ax.set_xlabel('Class', fontsize=20)
    ax.set_ylabel('Samples', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.savefig('res/Samples.pdf', format='pdf', dpi=1200)

