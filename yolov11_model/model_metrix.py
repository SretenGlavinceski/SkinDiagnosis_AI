from ultralytics import YOLO

model = YOLO('best.pt')

metrics = model.val(data="../dataset/data.yaml")
print(metrics)
