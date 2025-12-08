from pathlib import Path

def print_header():
    header_path = Path(__file__).resolve().parent.parent / "data" / "header.txt"
    try:
        with header_path.open("r", encoding="utf-8") as fh:
            print(fh.read())
    except FileNotFoundError:
        print(f"header file not found: {header_path}")