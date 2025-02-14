from ultralytics import YOLO


def predict_image(image_path):
    model = YOLO("C:/Users/glavi/PythonExcercise/ai_computer_vision/yolov11_model/best.pt")

    results = model.predict(source=image_path, show=True)

    # 0: 'benign', 1: 'melanoma'
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_idx = int(box.cls.item())
            confidence = box.conf.item()
            coordinates = box.xyxy.cpu().numpy().tolist()
            detections.append({
                "label": class_idx,
                "confidence": confidence,
                "coordinates": coordinates
            })

    return detections


# TEST
# image_path = "../dataset/train/images/benign--13-_jpg.rf.3ae13f0847b4e7d5661dd9af258f7fa6.jpg"
# print(predict_image('best.pt', image_path))
