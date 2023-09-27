import cv2

image = cv2.imread('data/test1.png')
_, image = cv2.imencode('.jpg', image)
image = image.tobytes()