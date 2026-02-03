import argparse
import shutil
import time
import struct
import socket
from pathlib import Path
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QMutex
from PyQt5.QtWidgets import QApplication, QMainWindow
import ui  #ui文件
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
# from utils.aws.resume import opt
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from PIL import Image
import sys
# from qt_material import apply_stylesheet
import qdarkstyle
import measure
import preProcess
time_total=time.time()

flag=1#运行标志
soc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)    # 使用ipv4 tcp协议
# soc.connect(('192.168.13.251', 6666)) # 服务器地址
soc.connect(('10.132.20.204', 6666))
def _open(filepath):    # 传输文件打开路径并读取数据
    file = open(filepath, 'rb')
    return file.read()


app = QApplication(sys.argv)
class Thread1(QThread):        # 线程1
    thread1_signal2 = pyqtSignal(object)
    def __init__(self):
        super(Thread1, self).__init__()
        self.Image_thread1 = None
        print('Start Model Initial!')

        #加载模型参数及数据集
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=r'best2.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default=r'/home/jetson/yolo/yolov5-5.0/data/capread/color_images', help='source')  # file/folder, 0 for webcam
        # parser.add_argument('--source', type=str, default=r"0",help='source')  # file/folder, 0 for webcam
        # 摄像头source为0，datasets.py里280line也要改
        parser.add_argument('--img-size', type=int, default=800, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        print(self.opt)
        check_requirements(exclude=('pycocotools', 'thop'))
        self.source, self.weights, self.view_img, self.save_txt, self.imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        self.save_img = not self.opt.nosave and not self.source.endswith('.txt')  # save inference images
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        #删除上次结果保存文件夹
        if os.path.exists('/home/jetson/yolo/yolov5-5.0/runs/detect/exp'):
            shutil.rmtree('/home/jetson/yolo/yolov5-5.0/runs/detect/exp')
        # Directories
        self.save_dir = Path(
            increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok))  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(
                self.device).eval()

        # Set Dataloader
        self.vid_path, self.vid_writer = None, None
        if self.webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride)
        else:
            self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        print('Model Initial Success!!')

    def detect(self,save_img=True):
        # Run inference
        # if os.path.exists('./runs/detect/exp'):
        #     shutil.rmtree('./runs/detect/exp')
        t0 = time.time()
        print('Load Data!')
        # cap = cv2.VideoCapture(0)
        print('Start Capture!')
        # time.sleep(2)
        # for i in range(10):  # 模型初始化前拍10张图片并预处理
        #     ret, frame = cap.read()
        #     print(f'im{i} Over!', end=' ')
        #     cv2.imwrite(f'/home/industry/yolov5-5.0/data/capread/im{i}.jpg', frame)
        #     preProcess.preProcess(f'/home/industry/yolov5-5.0/data/capread/im{i}.jpg')
        color_img_path='/home/jetson/yolo/yolov5-5.0/data/capread/color_images'
        depth_img_path='/home/jetson/yolo/yolov5-5.0/data/capread/depth_images'

        script_path='/home/jetson/astra/pyorbbecsdk/examples/save_image_plus.py'
        os.system(f'python3 {script_path}')
        print('\nCapture Over!')
        print('Start Preprocess!')
        # preProcess.onlyTran(f'/home/jetson/yolo/yolov5-5.0/data/capread/color_images/color0.jpg',f'/home/jetson/yolo/yolov5-5.0/data/capread/color_images/box.txt')
        print('Preprocess OK!')

        if self.source!='0':
            self.vid_path, self.vid_writer = None, None
            if self.webcam:
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride)
            else:
                self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)
            # Get names and colors
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        print('Load Data Over!')
        print('Start Identify!')
        count = 0
        global flag
        for path, img, im0s, vid_cap in self.dataset:
            if flag:
                count += 1
                print(f'im{count} identifying..')
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = self.model(img, augment=self.opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)
                t2 = time_synchronized()

                # Apply Classifier
                if self.classify:
                    pred = apply_classifier(pred, self.modelc, img, im0s)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    # print('遍历结果')
                    if self.webcam:  # batch_size >= 1
                        p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), self.dataset.count
                    else:
                        p, s, im0, frame = path, '', im0s, getattr(self.dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(self.save_dir / p.name)  # img.jpg
                    txt_path = str(self.save_dir / 'labels' / p.stem) + (
                        '' if self.dataset.mode == 'image' else f'_{frame}')  # img.txt
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        # if count==20:#保存第20帧
                        #     cv2.imwrite('rst.jpg',im0)
                        for *xyxy, conf, cls in reversed(det):
                            if self.save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or self.view_img:  # Add bbox to image
                                ssssss='screw' if str(self.names[int(cls)])=='0' else 'shim'

                                label = f'{ssssss} {conf:.2f}'
                                plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)

                    print(f'{s}Done. ({t2 - t1:.3f}s)')

                    # Stream results
                    # if self.view_img:
                    #     cv2.imshow(str(p), im0)
                    #     cv2.waitKey(1)  # 1 millisecond
                    self.Image_thread1 = QPixmap.fromImage(Image.fromarray(im0).toqimage())
                    self.thread1_signal2.emit(self.Image_thread1)

                    if save_img:
                        if self.dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if self.vid_path != save_path:  # new video
                                self.vid_path = save_path
                                if isinstance(self.vid_writer, cv2.VideoWriter):
                                    self.vid_writer.release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path += '.mp4'
                                self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            self.vid_writer.write(im0)
            else:
                break

            if count > 30:
                flag = 0
        if self.save_txt or save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
            print(f"Results saved to {self.save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')

    def run(self):
        print('线程1启动')

        self.detect()
        path = r'/home/jetson/yolo/yolov5-5.0/runs/detect/exp/labels'  # 转换前txt保存的文件夹
        save = r'/home/jetson/Desktop/result_m'  # 转换后txt保存的文件夹
        if not os.path.exists(save):
            os.makedirs(save)
            file = open('/home/jetson/Desktop/result_m/WHUT-A+Moreplus-R3.txt','w')
            file.close()
        path_item = os.path.join(path, 'color0.txt')
        path_item2 = os.path.join(save, 'WHUT-A+Moreplus-R3.txt')  # 保存的路径
        print('Start Process!')
        measure.measure('/home/jetson/yolo/yolov5-5.0/data/capread/color_images/color0.jpg', path_item, path_item2)
        print('Process Over!')

        #裁判盒输出
        # global time_total
        # time_total=time.time()-time_total
        # if time_total<20:
        #     time.sleep(21-time_total)
        print('Send Result!')
        while (1):
            size = os.path.getsize(path_item2)  # 得到要传输的文件所要的数据包大小
            tx = struct.pack('!ii', 2, size)  # 定义包头 和 数据包大小
            time.sleep(1)
            soc.sendall(tx)  # 发送自定义数据包头和数据包大小
            filepath = path_item2  # 要发送的文件路径
            filedata = _open(filepath)  # 读取文件
            soc.send(filedata)  # 发送数据
            print('All Over!')
            break



class MyWindows(ui.Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(MyWindows, self).__init__()
        self.setupUi(self)

        self.preview_thread = Thread1()#创建线程，初始化模型

        self.textBrowser.append('模型准备就绪!')
        self.textBrowser.ensureCursorVisible()  # 文本框显示文字
        self.label_2.setPixmap(QPixmap('img.jpg'))

        self.pushButton.clicked.connect(self.start)
        self.pushButton_2.clicked.connect(self.stop)

        self.preview_thread.thread1_signal2.connect(self.showImg)

    def start(self):
        global flag
        flag=1
        self.textBrowser.clear()  # 清除所有文本
        self.textBrowser.append('识别中.....')
        self.textBrowser.ensureCursorVisible()  # 文本框显示文字

        self.preview_thread.start()#开启识别线程;
        # global time_total
        # time_total=time.time()
        #向裁判盒发送开始信号
        # try:
        #     soc.connect(('192.168.1.66', 6666))
        # except:
        #     pass
        tx_buf = struct.pack('!ii', 0, 10)  # (定义一个结构体，使用struct.pack,!ii表示传输两个变量，分别为INT型)   #分别为包头，数据包长度 自定义数据包
        soc.sendall(tx_buf)  # 向服务器发送 包头和数据包长度
        soc.sendall(bytes('A+Moreplus', encoding="utf-8"))
        print('信号发送完成')
        # self.textBrowser.clear()  # 清除所有文本
        # self.textBrowser.append('识别结束')
        # self.textBrowser.ensureCursorVisible()  # 文本框显示文字
    def showImg(self,img):#显示图像
        self.label_2.setPixmap(img)
        # print('显示图像')

    def stop(self):
        global flag
        flag=0
        self.textBrowser.clear()  # 清除所有文本
        self.textBrowser.append('停止!')
        self.textBrowser.ensureCursorVisible()  # 文本框显示文字

# （更改所有file_name 为之前生成的界面py的文件名，如name，不加.py）
if __name__ == '__main__':
    my_windows = MyWindows()
    # apply_stylesheet(app,theme='dark_teal.xml')
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    # my_windows.setFixedSize(680,750)
    my_windows.show()
    sys.exit(app.exec_())
