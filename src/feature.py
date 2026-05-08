from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_processing import main


if __name__ == "__main__":
    main()
