# OpenCV 图像处理示例

这是人脸识别与特效技术项目的基础 OpenCV 示例程序，完成以下流程：

1. 从命令行读取图片路径。
2. 使用 OpenCV 读取图片。
3. 将 BGR 彩色图像转换为灰度图。
4. 分别显示原图和灰度图。

## 环境安装

需要 Python 3 和 OpenCV：

```powershell
pip install opencv-python numpy
```

## 运行程序

```powershell
python main.py "图片完整路径"
```

例如：

```powershell
python main.py "C:\temp\input.jpg"
```

程序会打开 `Original` 和 `Grayscale` 两个窗口，按任意键即可关闭。

Windows 下如果图片路径包含中文导致 OpenCV 无法读取，建议先将图片复制到不含中文的路径，例如 `C:\temp\input.jpg`。

## 运行测试

```powershell
python -m unittest tests.test_main -v
```

测试覆盖灰度转换、图片读取失败和缺少命令行参数等情况。

## 文件说明

- `main.py`：OpenCV 图像处理程序。
- `tests/test_main.py`：自动化测试。
- `README.md`：项目说明和使用方法。
