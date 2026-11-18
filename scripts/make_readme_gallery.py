"""Gallery of README.md: one figure from each chapter, taken from the executed notebooks.

The figures are not redrawn here: they are copied out of the notebooks, so the gallery
always shows what the chapters show.  Output: docs/images/gallery_chNN.png, and the
checks figure of Chapter 1, docs/images/checks_ch01.png.

Run from the repository root:  python scripts/make_readme_gallery.py
"""
import base64
import glob
import json
import os

root = os.path.join(os.path.dirname(__file__), "..")
# chapter number -> which figure of the notebook to use (0 = first)
choice = {1: 1, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1}

for chapter, index in choice.items():
    path = glob.glob(os.path.join(root, "notebooks", "*", f"{chapter:02d}_*.ipynb"))[0]
    notebook = json.load(open(path, encoding="utf-8"))
    images = [output["data"]["image/png"] for cell in notebook["cells"] for output in cell.get("outputs", [])
              if "image/png" in output.get("data", {})]
    out = os.path.join(root, "docs", "images", f"gallery_ch{chapter:02d}.png")
    with open(out, "wb") as handle:
        handle.write(base64.b64decode(images[index]))
    print(f"chapter {chapter}: figure {index + 1} of {len(images)} -> {os.path.normpath(out)}")

# the checks figure of Chapter 1 for the section "What the checks show"
path = glob.glob(os.path.join(root, "notebooks", "*", "01_*.ipynb"))[0]
notebook = json.load(open(path, encoding="utf-8"))
images = [output["data"]["image/png"] for cell in notebook["cells"] for output in cell.get("outputs", [])
          if "image/png" in output.get("data", {})]
with open(os.path.join(root, "docs", "images", "checks_ch01.png"), "wb") as handle:
    handle.write(base64.b64decode(images[-1]))
print("chapter 1: last figure -> docs/images/checks_ch01.png")
