#!/usr/bin/env python3

import os, glob, shutil

ext = "png"

dirname = "./sorted/"
if os.path.isdir(dirname):
    for file in glob.glob(dirname + f"*.png"):
        os.remove(file)
else:
    os.makedirs(dirname)

def parts(file):
    all_parts = file.split("_")
    prefix = "_".join(all_parts[:-1])
    field = all_parts[-1].split("_")[-1][1:-4]
    return prefix, field

def sort_val(file):
    return float(parts(file)[-1])

prefixes = set([ parts(file)[0] for file in glob.glob(f"*.{ext}") ])

for prefix in prefixes:
    files = glob.glob(f"{prefix}_h*.{ext}")
    files = sorted(files, key = sort_val)
    for idx, file in enumerate(files):
        hh = parts(file)[-1]
        new_file = file.replace(f"h{hh}", f"h{idx:03d}")
        shutil.copyfile(file, dirname + new_file)
