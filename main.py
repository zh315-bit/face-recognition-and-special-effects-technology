"""Read an image, convert it to grayscale, and display both versions."""

import sys

# 尝试导入 OpenCV；未安装时在运行阶段给出明确提示。
try:
    import cv2
except ImportError:
    cv2 = None


def load_image(image_path: str):
    """Load an image from disk or raise a clear error."""
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Run: pip install opencv-python")

    # imread 成功时返回 BGR 格式的三通道图像，失败时返回 None。
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def to_grayscale(image):
    """Convert a BGR image to a single-channel grayscale image."""
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Run: pip install opencv-python")

    # OpenCV 默认读取的是 BGR 图像，这里将其转换为单通道灰度图。
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def show_images(original, grayscale) -> None:
    """Display the original and grayscale images until a key is pressed."""
    # 分别创建原图和灰度图窗口，按任意键后释放全部窗口资源。
    cv2.imshow("Original", original)
    cv2.imshow("Grayscale", grayscale)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(arguments: list[str]) -> int:
    """Run the command-line image-processing workflow."""
    # 程序只接收一个图片路径参数。
    if len(arguments) != 1:
        print("Usage: python main.py <image-path>")
        return 2

    try:
        original = load_image(arguments[0])
    except (RuntimeError, ValueError) as error:
        # 将依赖缺失或读取失败的原因反馈到终端。
        print(error)
        return 1

    # 完成灰度化后显示两种图像。
    show_images(original, to_grayscale(original))
    return 0


if __name__ == "__main__":
    # 仅在直接运行脚本时读取命令行参数并返回程序退出码。
    raise SystemExit(main(sys.argv[1:]))
