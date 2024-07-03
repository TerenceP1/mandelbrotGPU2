import os
try:
    import PIL
except ImportError:
    print("No pillow library found!")
    os.system("python -m pip install pillow")

try:
    import cv2
except ImportError:
    print("No OpenCV library found!")
    os.system("python -m pip install opencv-python")