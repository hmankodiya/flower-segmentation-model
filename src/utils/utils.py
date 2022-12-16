import os
import shutil

def make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)