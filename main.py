import os
import sys

# Ensure src/ is in the module search path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import and execute main pipeline
from main import main

if __name__ == "__main__":
    main()

