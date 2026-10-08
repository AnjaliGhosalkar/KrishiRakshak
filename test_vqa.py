from __future__ import annotations

from pathlib import Path

from PIL import Image

from vqa_service import answer_question


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
QUESTION = "What is shown in this image?"


def main() -> None:
    image_dir = Path(__file__).resolve().parent / "test_images"
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported sample image found in {image_dir}")

    image_path = image_paths[0]
    with Image.open(image_path) as source_image:
        result = answer_question(source_image, QUESTION)

    print(f"Image: {image_path}")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
