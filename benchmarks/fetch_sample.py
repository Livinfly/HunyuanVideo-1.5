"""Fetch the pinned LongCat example; no dependency on another checkout."""
import argparse
from pathlib import Path
import urllib.request

from common import HERE, read_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'inputs')
    args = parser.parse_args()
    sample = read_json(HERE / 'sample.json')
    args.output.mkdir(parents=True, exist_ok=True)
    image = args.output / 'reference.png'
    if not image.exists() or sha256(image) != sample['image_sha256']:
        temporary = image.with_suffix('.download')
        with urllib.request.urlopen(sample['image_url'], timeout=120) as response:
            temporary.write_bytes(response.read())
        if sha256(temporary) != sample['image_sha256']:
            raise ValueError('Downloaded image checksum mismatch')
        temporary.replace(image)
    (args.output / 'prompt.txt').write_text(sample['prompt'] + '\n')
    print(image.resolve())


if __name__ == '__main__':
    main()
