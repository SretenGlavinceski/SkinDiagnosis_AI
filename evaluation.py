import os
from main import predict_image_info

total_images = 0
correct_predictions = {
    'YOLOv11': 0,
    'Naive-Bayes': 0,
    'Decision Tree': 0,
    'Random Forest': 0,
    'Neural Network': 0,
}

checkpoints = [10, 50, 100, 250, 500]
graph_info = {}

dataset_dir = 'dataset/train/images'

for image_name in os.listdir(dataset_dir):

    image_path = os.path.join(dataset_dir, image_name)
    total_images += 1
    results = predict_image_info(image_path)

    # print(results)

    for model_name, predicted_label in results.items():
        if predicted_label is not None and predicted_label.lower() in image_path.lower():
            correct_predictions[model_name] += 1

    if total_images in checkpoints:
        graph_info[total_images] = correct_predictions.copy()
        print(graph_info)
        if total_images == 500:
            break

# for model_name, correct_count in correct_predictions.items():
#     precision = correct_count / total_images
#     print(f"Precision for {model_name}: {precision:.2%}")

print(correct_predictions)
print(graph_info)


### OUTPUT:

# {10: {'YOLOv11': 10, 'Naive-Bayes': 10, 'Decision Tree': 10, 'Random Forest': 10, 'Neural Network': 10},
# 50: {'YOLOv11': 48, 'Naive-Bayes': 44, 'Decision Tree': 35, 'Random Forest': 43, 'Neural Network': 46},
# 100: {'YOLOv11': 96, 'Naive-Bayes': 90, 'Decision Tree': 72, 'Random Forest': 90, 'Neural Network': 95},
# 250: {'YOLOv11': 239, 'Naive-Bayes': 230, 'Decision Tree': 171, 'Random Forest': 231, 'Neural Network': 243},
# 500: {'YOLOv11': 483, 'Naive-Bayes': 435, 'Decision Tree': 369, 'Random Forest': 470, 'Neural Network': 486}}
