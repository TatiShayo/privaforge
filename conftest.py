# Ensures `import privaforge.*` resolves when pytest runs from inside the
# project directory (the package lives at the repo root's parent level).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
