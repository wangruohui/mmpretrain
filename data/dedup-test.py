"""Dedup test images from training images based on phash."""

from imagededup.methods import PHash
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainlist", type=Path, default="huangfan-shumei-1120-train.txt"
    )
    parser.add_argument("--trainroot", type=Path, default=".")
    parser.add_argument("--testlist", type=Path, default="dana1/ann_dana.txt")
    parser.add_argument("--testroot", type=Path, default="涉政与色情低俗图包")
    parser.add_argument("--duplist", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    if not args.output:
        name = args.trainlist.stem
        args.output = args.trainlist.with_stem(f"{name}-dedup")
    if not args.duplist:
        name = args.trainlist.stem
        args.duplist = args.trainlist.with_stem(f"{name}-dup")

    assert args.trainlist != args.output
    assert args.testlist != args.output

    return args


def main(args):
    # read test images and build phash index

    dup_count = 0

    test_images = {}
    with open(args.testlist) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            imfile = line.rsplit(" ", 1)[0]
            phash_file = str(imfile) + ".phash.txt"

            imfile = Path(args.testroot) / imfile
            phash_file = Path(args.testroot) / (phash_file)

            assert (imfile).exists(), imfile
            if not phash_file.exists():
                print(f"Warning: {phash_file} does not exist")
                continue
            phash = phash_file.read_text().strip()
            test_images[phash] = imfile

    with open(args.output, "w") as fo:
        with open(args.duplist, "w") as fdu:
            with open(args.trainlist) as f:
                for i, line in enumerate(tqdm(f)):
                    line = line.strip()
                    if not line:
                        continue
                    # print(line)
                    imfile, label = line.rsplit(" ", 1)
                    phash_file = str(imfile) + ".phash.txt"

                    imfile = Path(args.trainroot) / imfile
                    phash_file = Path(args.trainroot) / (phash_file)

                    if not imfile.exists():
                        print(f"Warning: {imfile} does not exist")
                        continue

                    if not phash_file.exists():
                        print(f"Warning: {phash_file} does not exist")
                        phash = PHash().encode_image(str(imfile))
                        print(phash)
                        exit()
                        phash_file.write_text(phash)
                    else:
                        phash = phash_file.read_text().strip()

                    if phash in test_images:
                        print(f"Found duplicate: {imfile} {test_images[phash]}")
                        dup_count += 1
                        print(f"Found {dup_count} duplicates")
                        fdu.write(f"{imfile} {test_images[phash]}\n")
                    else:
                        fo.write(f"{imfile} {label}\n")

    print("Save deduped trainlist to ", args.output)
    print("Save duplicated to ", args.duplist)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
