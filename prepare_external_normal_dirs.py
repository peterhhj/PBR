import os
import shutil
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict
from typing import Iterable
from typing import List


DEFAULT_VARIANTS = [
    "normal",
    "normal_from_depth",
    "fused_normal",
    "normal_on_bg",
]


def normalize_view_name(filename: str, variant: str) -> str:
    suffix = f"_{variant}"
    stem, ext = os.path.splitext(os.path.basename(filename))
    if not stem.endswith(suffix):
        raise ValueError(f"File '{filename}' does not match variant suffix '{suffix}'.")
    return stem[: -len(suffix)] + ext


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def collect_variant_files(input_dir: str, variants: Iterable[str]) -> Dict[str, List[str]]:
    files_by_variant: Dict[str, List[str]] = defaultdict(list)
    for entry in os.listdir(input_dir):
        source_path = os.path.join(input_dir, entry)
        if not os.path.isfile(source_path):
            continue
        for variant in variants:
            if entry.endswith(f"_{variant}.png") or entry.endswith(f"_{variant}.npy"):
                files_by_variant[variant].append(source_path)
                break
    return files_by_variant


def export_variant_directories(
    input_dir: str,
    output_root: str,
    variants: Iterable[str],
) -> None:
    ensure_dir(output_root)
    files_by_variant = collect_variant_files(input_dir, variants)

    if not files_by_variant:
        raise RuntimeError(
            f"No variant normal files were found in {input_dir}. "
            f"Expected names like 0000_normal.png or 0000_fused_normal.png."
        )

    for variant in variants:
        variant_output_dir = os.path.join(output_root, variant)
        ensure_dir(variant_output_dir)
        exported = 0

        for source_path in sorted(files_by_variant.get(variant, [])):
            target_name = normalize_view_name(source_path, variant)
            target_path = os.path.join(variant_output_dir, target_name)
            shutil.copy2(source_path, target_path)
            exported += 1

        print(f"[{variant}] exported {exported} files to {variant_output_dir}")


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Split dataset_normals-style files into per-variant directories for external normal training. "
            "Files are renamed from '<view>_<variant>.png' to '<view>.png'."
        )
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory like dataset_normals containing *_normal.png files")
    parser.add_argument("--output_root", type=str, required=True, help="Output root; subdirectories will be created per normal variant")
    parser.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=DEFAULT_VARIANTS,
        help=(
            "Normal variants to extract. "
            f"Default: {' '.join(DEFAULT_VARIANTS)}"
        ),
    )
    args = parser.parse_args()

    export_variant_directories(
        input_dir=args.input_dir,
        output_root=args.output_root,
        variants=args.variants,
    )


if __name__ == "__main__":
    main()
