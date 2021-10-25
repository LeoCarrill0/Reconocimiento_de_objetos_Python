import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
#from imutils.video import VideoStream
import time

MIN_MATCH_COUNT = 5
sift = cv2.xfeatures2d.SIFT_create()
#sift = cv2.xfeatures2d_SIFT
img1 = cv2.imread("20200615_130508(Box).jpg", 0)
img1 = imutils.resize(img1, width=200)
kp1, des1 = sift.detectAndCompute(img1, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
flan_n = cv2.FlannBasedMatcher(index_params, search_params)

vs = cv2.VideoCapture(0) #VideoStream(src=0).start()
time.sleep(2.0)

while True:
    _, frame = vs.read()
    img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = flan_n.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    matchesMask = 0
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    if matchesMask != 0:
        draw_params = dict(matchColor=(255, 0, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    cv2.imshow('gray', img3)

    key = cv2.waitKey(50) & 0xFF
    if key == ord("q"):
        break
