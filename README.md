# 中国机器人大赛-先进视觉代码

面向先进视觉赛题的视觉检测与测量系统。项目基于 YOLOv5 目标检测 + 传统几何测量流程，集成 PyQt5 界面、摄像头采集、结果解析与裁判盒 TCP 通信，可用于螺钉/垫片等零部件的识别与测量。

## 项目亮点
- 目标检测：基于 YOLOv5 的零部件识别与标注输出
- 测量流程：对检测结果进行尺寸/圆形测量（含 R3/R4 版本）
- 工业链路：集成采集、预处理、推理、测量、裁判盒通信的完整流程
- GUI 展示：PyQt5 界面实时展示识别结果

## 目录结构
- src/
  - detectFinal-b.py：R3 版本主流程（检测 + 测量 + 通信）
  - detectFinal-b-R4.py：R4 版本主流程（检测 + 圆形测量 + 通信）
  - save_image_plus.py：采集脚本（彩图/深度图）
  - save_image_plus_circle.py：采集脚本（圆形场景）
  - utlis.py：图像轮廓与几何辅助函数

## 运行环境
- Python 3.7+（建议 3.8/3.9）
- 依赖库（核心）：
  - PyQt5
  - qdarkstyle
  - OpenCV (cv2)
  - torch
  - numpy
  - Pillow
- 外部依赖：
  - YOLOv5 代码与模型权重
  - 采集脚本依赖的相机 SDK/驱动（如 Orbbec/Astra，按实际硬件调整）

## 快速开始
1. 准备 YOLOv5 工程与权重
   - 将 YOLOv5 项目放置在本机合适路径
   - 准备训练好的权重文件（如 best2.pt）
2. 配置采集与数据路径
   - 修改 detectFinal-b.py、detectFinal-b-R4.py 中的模型权重路径、数据目录、采集脚本路径
   - 确认采集脚本能生成 color_images/depth_images
3. 启动 GUI
   - 运行 src/detectFinal-b.py（R3）或 src/detectFinal-b-R4.py（R4）
   - 点击界面按钮开始检测

## 关键流程说明
1. 采集：调用采集脚本生成彩色/深度图
2. 检测：YOLOv5 推理生成检测框及置信度
3. 测量：将检测结果转换为测量数据并生成输出文件
4. 通信：TCP 将结果发送至裁判盒
