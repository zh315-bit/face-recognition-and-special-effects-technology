"""Read an image, convert it to grayscale, and display both versions."""

import sys

try:
    import cv2
except ImportError:
    cv2 = None


def load_image(image_path: str):
    """Load an image from disk or raise a clear error."""
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Run: pip install opencv-python")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def to_grayscale(image):
    """Convert a BGR image to a single-channel grayscale image."""
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Run: pip install opencv-python")

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def show_images(original, grayscale) -> None:
    """Display the original and grayscale images until a key is pressed."""
    cv2.imshow("Original", original)
    cv2.imshow("Grayscale", grayscale)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(arguments: list[str]) -> int:
    """Run the command-line image-processing workflow."""
    if len(arguments) != 1:
        print("Usage: python main.py <image-path>")
        return 2

    try:
        original = load_image(arguments[0])
    except (RuntimeError, ValueError) as error:
        print(error)
        return 1

    show_images(original, to_grayscale(original))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
