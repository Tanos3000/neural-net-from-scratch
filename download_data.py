"""Downloads the raw MNIST dataset (handwritten digit images) into data/.

Source: Google's public mirror of the original MNIST files (same files
Yann LeCun's site serves, just on a faster/more reliable host). Public domain.
"""

import ssl
from pathlib import Path
from urllib.request import urlopen

import certifi

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]
DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())

    for filename in FILES:
        target = DATA_DIR / filename
        if target.exists():
            print(f"Already downloaded: {target}")
            continue

        print(f"Downloading {filename}...")
        with urlopen(BASE_URL + filename, context=context) as response:
            target.write_bytes(response.read())

    print("Done.")


if __name__ == "__main__":
    main()
