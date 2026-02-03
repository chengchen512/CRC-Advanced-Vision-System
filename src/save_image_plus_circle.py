# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.  
#  You may obtain a copy of the License at
#  
#      http:# www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
from tkinter.tix import ButtonBox
import cv2
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import Config
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import OBError
from pyorbbecsdk import VideoStreamProfile

import numpy as np
from utils import frame_to_bgr_image
import time

ESC_KEY = 27


def reorder(myPoints):#对角点重新排序
    w_c = 320
    h_c = 240
    h_bias = -20
    w_bias = 0
    w_scale = 2.35
    h_scale = 2.3
    myPoints = myPoints.reshape((4, 2))
    myPoints[0][0] = ((myPoints[0][0] - w_c) * w_scale + 960).astype('int') + w_bias
    myPoints[0][1] = ((myPoints[0][1] - h_c) * h_scale + 540).astype('int') + h_bias
    myPoints[1][0] = ((myPoints[1][0] - w_c) * w_scale + 960).astype('int') + w_bias
    myPoints[1][1] = ((myPoints[1][1] - h_c) * h_scale + 540).astype('int') + h_bias
    myPoints[2][0] = ((myPoints[2][0] - w_c) * w_scale + 960).astype('int') + w_bias
    myPoints[2][1] = ((myPoints[2][1] - h_c) * h_scale + 540).astype('int') + h_bias
    myPoints[3][0] = ((myPoints[3][0] - w_c) * w_scale + 960).astype('int') + w_bias
    myPoints[3][1] = ((myPoints[3][1] - h_c) * h_scale + 540).astype('int') + h_bias
    
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def warpImg(img, points, w, h, pad=0):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    imgWarp = imgWarp[pad:imgWarp.shape[0] - pad, pad:imgWarp.shape[1] - pad]
    return imgWarp


def findbox(depth_frame):
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()

    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_data = depth_data.reshape((height, width))
    depth_data = depth_data.astype(np.float32) * scale
    depth_data = np.where((depth_data > 1750), 3000, depth_data)
    depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # ret, th = cv2.threshold(depth_image, 0, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
    # cv2.imshow('depth', depth_image)

    depth_image = cv2.GaussianBlur(depth_image, (5, 5), 0)
    canny = cv2.Canny(depth_image, 50, 100)
    kernel = np.ones((5, 5))
    canny = cv2.dilate(canny, kernel, iterations=10)  # 膨胀
    canny = cv2.erode(canny, kernel, iterations=10)  # 腐蚀

    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('depth_canny', canny)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if sorted_contours[1] is not None:
        max_contour = sorted_contours[1]
    else:
        return

    x, y, w, h = cv2.boundingRect(max_contour)
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y
    x3 = x + w
    y3 = y + h
    x4 = x
    y4 = y + h

    box = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

    print(box)
    # box_img = np.zeros((480, 640, 3), dtype=np.int8)
    # box_img = cv2.rectangle(box_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv2.drawContours(box_img, max_contour, -1, (0, 255, 0), 2)
    # cv2.imshow('box', box_img)
    return box


def main():
    pipeline = Pipeline()
    config = Config()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_video_stream_profile(1920, 1080, OBFormat.RGB, 10)
        config.enable_stream(color_profile)
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        assert profile_list is not None
        depth_profile = profile_list.get_video_stream_profile(640, 480, OBFormat.Y16, 10)
        assert depth_profile is not None
        print("color profile : {}x{}@{}_{}".format(color_profile.get_width(),
                                                   color_profile.get_height(),
                                                   color_profile.get_fps(),
                                                   color_profile.get_format()))
        print("depth profile : {}x{}@{}_{}".format(depth_profile.get_width(),
                                                   depth_profile.get_height(),
                                                   depth_profile.get_fps(),
                                                   depth_profile.get_format()))
        config.enable_stream(depth_profile)
    except Exception as e:
        print(e)
        return
    try:
        pipeline.start(config)
    except Exception as e:
        print(e)
        return

    failed_cnt = 0
    while True:
        try:
            time.sleep(0.2)
            frames: FrameSet = pipeline.wait_for_frames(500)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            # covert to RGB format
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue
            # cv2.imshow("Color Viewer", color_image)
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                print('depth frame is None')
                continue
            box = findbox(depth_frame)
            if box is None:
                print('+++++++++++++++++box not found retry:')
                print(failed_cnt)
                failed_cnt = failed_cnt+1
                continue
            box = reorder(box)
            print(box)
            # color_image = cv2.rectangle(color_image, box[0], box[3], (0, 255, 0), 2)
            color_image = warpImg(color_image, box, 1920, 1080, 0)
            color_image_640 = cv2.resize(color_image, (640, 480))
            # box = box.reshape(8, 1)
            # store_str = ''
            # for point in box:
            #     print(point)
            #     store_str = store_str+str(point)[1:-1]+' '
            # store_str = store_str[:-1]
            # print('------------------------------------')
            # #将box保存在box.txt文件下
            # with open("/home/jetson/yolo/yolov5-5.0/data/capread/color_images/box.txt", "w") as f:
            #     f.write(store_str)
            #将图像保存在capread/colorimage文件夹中
            color_image_640 = cv2.cvtColor(color_image_640, cv2.COLOR_BGR2RGB)
            cv2.imwrite('/home/jetson/yolo/yolov5-5.0/data/capread/color_images/color0.jpg', color_image_640)
            # cv2.imwrite('/home/jetson/yolo/yolov5-5.0/data/capread/depth_images/depth0.raw', depth_image)
            break
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break
        except KeyboardInterrupt:
            break
    pipeline.stop()


if __name__ == "__main__":
    main()
    # color = cv2.imread('/home/jetson/yolo/yolov5-5.0/data/capread/color_images/color0.jpg')
    # cv2.imshow('cut', color)
    # cv2.waitKey(1)
    # time.sleep(10)
