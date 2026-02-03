import cv2
import numpy as np

def getContours(img,cThr=[50,70],showCanny=True,minArea=1000,filter=0,draw =False):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('test',imgGray)
    # imgGray=img
    imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
    imgCanny = cv2.Canny(imgBlur,cThr[0],cThr[1])#当函数 cv2.Canny()的参数 threshold1 和 threshold2 的值较小时，能够捕获更多的边缘信息。
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)#膨胀
    imgThre = cv2.erode(imgDial,kernel,iterations=2)#腐蚀
    if showCanny:cv2.imshow('Canny',imgThre)
    binary,contours,hiearchy = cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i,True)#周长
            approx = cv2.approxPolyDP(i,0.02*peri,True)#轮廓角点坐标
            #print(len(approx))
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx),area,approx,bbox,i])
            else:
                finalCountours.append([len(approx),area,approx,bbox,i])
    finalCountours = sorted(finalCountours,key = lambda x:x[1] ,reverse= True)#对所有轮廓进行排序
    if draw:
        for con in finalCountours:
            cv2.drawContours(img,con[4],-1,(0,0,255),3)
    return img, finalCountours

def reorder(myPoints):#对角点重新排序
    #print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def warpImg_v2(img,  h, w, pad=5):
    # print(points)
    # 定义膨胀的核（矩形核）
    kernel = np.ones((5, 5), np.uint8)
    src_img = img
    imgWarp = None
    # src_img = cv2.resize(src_img, (1280, 720))
    # cv2.imshow('src', src_img)
    # 将图像转换为灰度
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    # 使用Canny边缘检测
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, threshold1=50, threshold2=70)
    # 寻找轮廓
    _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(src_img)
    # 遍历所有轮廓，仅使用最大轮廓
    for contour in contours:
        if cv2.contourArea(contour) > 2000:
            cv2.drawContours(contour_image, contour, -1, (0, 255, 0), 2)
            ellipse = cv2.fitEllipse(contour)
            # 从拟合结果中获取椭圆参数
            center, axes, angle = ellipse
            major_axis, minor_axis = axes
            # 在图像上绘制椭圆
            image_with_ellipse = np.zeros_like(src_img)
            cv2.ellipse(image_with_ellipse, ellipse, (0, 255, 0), 2)

            # 输出椭圆的参数
            print("椭圆中心:", center)
            print("长轴:", major_axis)
            print("短轴:", minor_axis)
            print("旋转角度:", angle)

            # 计算外接矩形的四个角点
            rect_pts = cv2.boxPoints((center, axes, angle))  # 获取外接矩形的四个顶点
            rect_pts = np.int0(rect_pts)  # 将浮点数坐标转换为整数坐标
            rect_pts = reorder(rect_pts)
            print(rect_pts)
            points = rect_pts
            pts1 = np.float32(points)
            pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarp = cv2.warpPerspective(img, matrix, (w, h))
            imgWarp = imgWarp[pad:imgWarp.shape[0] - pad, pad:imgWarp.shape[1] - pad]

    return imgWarp

def warpImg (img, points, w, h, pad=5):
     #print(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad]
    return imgWarp

def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5


