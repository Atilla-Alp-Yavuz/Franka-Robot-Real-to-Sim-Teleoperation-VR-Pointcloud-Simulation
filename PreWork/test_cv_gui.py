# test_cv_gui_ok.py
import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)

# Make a simple black image
img = np.zeros((480, 640, 3), dtype=np.uint8)
img[:] = (0, 0, 255)  # red

cv2.imshow("test", img)
print("Press any key in the image window to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()