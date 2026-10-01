"""Copy existing project images into a portfolio; do not regenerate results."""
from pathlib import Path
import argparse
import shutil


def copy_images(source, destination):
    if not source.exists():
        print(f"Not found: {source}")
        return 0
    count = 0
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--site-root", type=Path, required=True)
    # Folder labels are assumptions in the draft; map your actual directories here.
    parser.add_argument("--hybrid2-dir", default="outputs/gigachad_fatalis")
    parser.add_argument("--hybrid3-dir", default="outputs/beatles_kendrick")
    args = parser.parse_args()
    source = args.project_root.resolve()
    target = args.site_root.resolve() / "assets/images/project2"
    if target.is_relative_to(source):
        raise ValueError("Keep the portfolio asset destination outside the source project to avoid recursive copying.")
    count = sum(copy_images(source / folder, target / folder) for folder in ["img", "out", "outputs"])
    # Map your real custom folders to the paths expected by the draft HTML.
    for actual, expected in [(args.hybrid2_dir, "outputs/gigachad_fatalis"),
                             (args.hybrid3_dir, "outputs/beatles_kendrick")]:
        if actual != expected:
            count += copy_images(source / actual, target / expected)
    print(f"Copied {count} image files to {target}")


if __name__ == "__main__":
    main()
