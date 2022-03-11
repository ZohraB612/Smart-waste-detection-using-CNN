import os

from PIL import Image

image_path = r"C:/Users/Zohra/MScProject/faster_rcnn_and_ssd/waste/images"
old_size = image_path + "/old_size"

for img in os.listdir(old_size):
    im = Image.open(old_size + "/" + img)
    # image.show()

    # The file format of the source file.
    # print(image.format) # Output: JPEG

    # The pixel format used by the image. Typical values are "1", "L", "RGB", or "CMYK."
    # print(image.mode) # Output: RGB

    # Image size, in pixels. The size is given as a 2-tuple (width, height).
    # print(image.size) # Output: (512, 384)

    # Colour palette table, if any.
    # print(image.palette) # Output: None

    new_im = im.resize((224, 224))
    new_im.save(image_path + '/224x224' + "/" + img)

    print(im.size)  # Output: (512, 384)
    print(new_im.size)  # Output: (640, 640)
