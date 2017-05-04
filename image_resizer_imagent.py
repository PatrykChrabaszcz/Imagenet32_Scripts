from PIL import Image
from argparse import ArgumentParser
import os
from multiprocessing import Pool

alg_dict = {
    'lanczos': Image.LANCZOS,
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'hamming': Image.HAMMING,
    'box': Image.BOX
}


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-i', '--in_dir', help="Input directory with source images", required=True)
    parser.add_argument('-o', '--out_dir', help="Output directory for resized images", required=True)
    parser.add_argument('-s', '--size', help="Size of an output image (e.g. 32 results in (32x32) image)",
                        default=32, type=int)
    parser.add_argument('-a', '--algorithm', help="Algorithm used for resampling: lanczos, nearest,"
                                                  " bilinear, bicubic, box, hamming",
                        default='lanczos')

    parser.add_argument('-r', '--recurrent', help="Process all subfolders in this folder (1 lvl deep)",
                        action='store_true')
    parser.add_argument('-f', '--full', help="Use all algorithms, create subdirectory for each algorithm output",
                        action='store_true')
    parser.add_argument('-e', '--every_nth', help="Use if you don't want to take all classes, "
                                                  "if -e 10 then takes every 10th class",
                        default=1, type=int)
    parser.add_argument('-j', '--processes', help="Number of sub-processes that run different folders "
                                                  "in the same time ",
                        default=1, type=int)
    args = parser.parse_args()

    return args.in_dir, args.out_dir, args.algorithm, args.size, args.recurrent, \
           args.full, args.every_nth, args.processes


def str2alg(str):
    str = str.lower()
    return alg_dict.get(str, None)


# Takes in_dir, out_dir and alg as strings
# resize images from in_dir using algorithm deduced from
# alg string and puts them to "out_dir/alg/" folder
def resize_img_folder(in_dir, out_dir, alg):
    print('Folder %s' % in_dir)

    alg_val = str2alg(alg)

    if alg_val is None:
        print("Sorry but this algorithm (%s) is not available, use help for more info." % alg)
        return

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for filename in os.listdir(in_dir):
        # Exception raised when file is not an image
        try:
            im = Image.open(os.path.join(in_dir, filename))

            # Convert grayscale images into 3 channels
            if im.mode != "RGB":
                im = im.convert(mode="RGB")

            im_resized = im.resize((size, size), alg_val)
            # Get rid of extension (.jpg or other)
            filename = os.path.splitext(filename)[0]
            im_resized.save(os.path.join(out_dir, filename + '.png'))
        except OSError as err:
            print("This file couldn't be read as an image")
            with open("log.txt", "a") as f:
                f.write("Couldn't resize: %s" % os.path.join(in_dir, filename))


if __name__ == '__main__':
    in_dir, out_dir, alg_str, size, recurrent, full, every_nth, processes = parse_arguments()

    print('Starting ...')

    if full is False:
        algs = [alg_str]
    else:
        algs = alg_dict.keys()

    pool = Pool(processes=processes)

    repeat = False
    for alg in algs:
        print('Using algorithm %s ...' % alg)
        current_out_dir = os.path.join(out_dir, alg)
        if recurrent:
            print('Recurrent for all folders in in_dir:\n %s' % in_dir)
            folders = [dir for dir in sorted(os.listdir(in_dir)) if os.path.isdir(os.path.join(in_dir, dir))]
            for i, folder in enumerate(folders):
                if i % every_nth is 0 or repeat is True:
                    r = pool.apply_async(
                        func=resize_img_folder,
                        args=[os.path.join(in_dir, folder), os.path.join(current_out_dir, folder), alg])

        else:
            print('For folder %s' % in_dir)
            resize_img_folder(in_dir=in_dir, out_dir=current_out_dir, alg=alg)
    pool.close()
    pool.join()
    print("Finished.")
