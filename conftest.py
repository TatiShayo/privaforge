# Ensures `import privaforge.*` and direct imports resolve hermetically
import os
import sys

pkg_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(pkg_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)
