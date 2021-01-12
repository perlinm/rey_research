#!/usr/bin/env python3

import os, glob, PIL.Image

ext = "png"

dirname = "./sorted/"
if os.path.isdir(dirname):
    for file in glob.glob(dirname + f"*.{ext}"):
        os.remove(file)
else:
    os.makedirs(dirname)

# vertically concatenate images
def get_concat_v(*images):
    width = images[0].width
    heights = [ sum([ image.height for image in images[:ii] ])
                for ii in range(len(images)+1) ]
    comb = PIL.Image.new("RGB", (width, heights[-1]))
    for image, pos in zip(images, heights):
        comb.paste(image, (0, pos))
    return comb

# get file tags
def get_tags(file):
    parts = os.path.basename(file).split("_")[-2:] # assumes two tags
    parts[-1] = ".".join(parts[-1].split(".")[:-1])
    return { part[0] : part[1:] for part in parts }

# get qudit dimensions
dims = set([ get_tags(file)["d"] for file in glob.glob(f"*.{ext}") ])

def sort_val(file):
    return float(get_tags(file)["h"])

for dim in dims:
    files = glob.glob(f"bloch_d{dim}_h*.{ext}")
    files = sorted(files, key = sort_val)
    for idx, file in enumerate(files):
        hh = get_tags(file)["h"]
        new_file = file.replace(f"h{hh}", f"h{idx:03d}")

        img1 = PIL.Image.open(file)
        img2 = PIL.Image.open(file.replace("bloch","top"))
        img3 = PIL.Image.open(file.replace("bloch","side"))
        get_concat_v(img1, img2, img3).save(dirname + new_file)
