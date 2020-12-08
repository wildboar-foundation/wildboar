import os
import sys

sys.path.insert(1, os.path.abspath("code/"))


if __name__ == "__main__":
    img_path = os.path.abspath("_static/img")
    import img

    if not os.path.exists(img_path):
        os.mkdir(img_path)

    img.build_all(img_path, img.TUTORIAL, subdir="tutorial")
    img.build_all(img_path, img.COUNTERFACTUALS, subdir="explain")
