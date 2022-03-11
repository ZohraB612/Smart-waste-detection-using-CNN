from PIL import Image
from imutils import build_montages, paths
import argparse
import random
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to input directory of images")
ap.add_argument("-s", "--sample", type=int, default=21,
                help="# of images to sample")
args = vars(ap.parse_args())

# grab the paths to the images, then randomly select a sample of
# them
imagePaths = list(paths.list_images(args["images"]))
random.shuffle(imagePaths)
# imagePaths = imagePaths[:args["sample"]]

# initialize the list of images
images = []

# loop over the list of image paths
for imagePath in imagePaths:
    # load the image and update the list of images
    image = cv2.imread(imagePath)
    images.append(image)

# construct the montages for the images
# montages = build_montages(images, (128, 196), (3, 2))
montages = build_montages(images, (196, 196), (2, 2))

# loop over the montages and display each of them
c = 1
for montage in montages:
    print(type(montage))
    im = Image.fromarray(montage)
    folder = r"C:/Users/Zohra/MScProject/faster_rcnn_and_ssd/waste/images/image-collage/new_collage_4/"
    im.save(folder + "collage" + str(c) + ".png")
    # cv2.imshow("Montage", montage)
    c += 1
    cv2.waitKey(0)
