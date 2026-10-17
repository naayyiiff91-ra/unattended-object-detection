import os
import sys

# Support running 'python src/main.py' even if the terminal is already inside src/
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from main import main

if __name__ == "__main__":
    main()

