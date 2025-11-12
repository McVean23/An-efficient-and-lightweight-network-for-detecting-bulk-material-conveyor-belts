from ultralytics import YOLO

# Create a new YOLO model from scratch  选择一个训练模型文件，重新开始训练
model = YOLO('/home/tq/pycharm_projects/YOLOv8/ultralytics-main/ultralytics/cfg/models/v8/yolov8-seg-head-2025-1108.yaml')  #YOLOv8模型都有，不需要自己写，该路经就行
# model = YOLO('/home/tq/pycharm_projects/YOLOv8/ultralytics-main/ultralytics/cfg/models/v8/yolov8.yaml')
# # Load a pretrained YOLO model (recommended for training)  这里是加载预训练模型，重新开始训练注释掉
# model = YOLO('/home/tq/pycharm_projects/YOLOv8/ultralytics-main/runs/segment/arrester_20240112/weights/arrester_best_20240112.pt')

# Train the model using the 'coco128.yaml' dataset for 300 epochs
results = model.train(data='/home/tq/dataset/belt.v24i.yolov8/data.yaml', epochs=100, imgsz=640, batch=16)

# Evaluate the model's performance on the validation set
# results = model.val()

# # Perform object detection on an images using the model
# results = model('https://ultralytics.com/images/bus.jpg')
#
# # Export the model to ONNX format
# success = model.export(format='onnx')wo