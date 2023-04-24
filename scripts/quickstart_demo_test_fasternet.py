from ultralytics import YOLO
from custom_tools.cvutils import read_image
from pathlib import Path

if __name__ == "__main__":
    model_yaml_path = r"D:\code\yolov8\ultralytics\ultralytics\models\v8\yolov8s_fasternet.yaml"
    # Create a new YOLO model from scratch
    model = YOLO(model_yaml_path)
    # model = YOLO("yolov8s.yaml")

    # # Load a pretrained YOLO model (recommended for training)
    # model = YOLO(r'D:\pretrained_model\yolov8\yolov8n.pt')

    # # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='coco128.yaml', epochs=3, workers=0, imgsz=320)

    # # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the model
    img_path = str(Path(__file__).parent.parent / "data" / "demo.jpg")
    img = read_image(img_path)
    results = model(img)
    # results = model('https://ultralytics.com/images/bus.jpg')

    # Export the model to ONNX format
    success = model.export(format='onnx')