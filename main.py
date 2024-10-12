from ultralytics import YOLO
import torch
if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    results = model.train(data='config.yaml', epochs=5, imgsz=640, device="cuda", batch=42)
    #metrics = model.val()  # evaluate model performance on the validation set
    print(results)