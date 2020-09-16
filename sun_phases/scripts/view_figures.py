#!/usr/bin/env python3

import os, sys

view_cmd = "pdf"
files = sys.argv[1:]

def val(file):
    try: return ( float(file.split("_")[-2][1:]), file )
    except: return file

sorted_files = sorted(files, key = val)

os.system(" ".join( [view_cmd] + sorted_files ))
