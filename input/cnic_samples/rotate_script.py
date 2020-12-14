import cv2
import imutils

import argparse
import os


def rotateImages(rotationAmt):
    # for each image in the current directory
    print(rotationAmt)
    for image in os.listdir(os.getcwd()):
        # open the image
        print(image)
        if image.endswith(".py"):
            continue
        img = cv2.imread(image)
        # rotate and save the image with the same filename
        rotated = imutils.rotate_bound(img, rotationAmt)
        # close the image
        name = f"_{rotationAmt}.png"
        cv2.imwrite(image+name, rotated)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pytho Script to rotate all images in given directory by specified angle"
    )

    parser.add_argument(
        "-a",
        "--angle",
        help="Angle of rotation",
        required=True,
        dest="angle",
    )

    args = parser.parse_args()

    rotation_angle = int(args.angle)
    print(rotation_angle)
    rotateImages(rotation_angle)
