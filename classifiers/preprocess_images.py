import os
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from feature_extraction import extract_features
import csv

folder_path_labels = '../dataset/valid/labels'
folder_path_images = '../dataset/valid/images'

with open('data_features.csv', mode='a', newline='') as file:
    writer = csv.writer(file)

    for file_name in os.listdir(folder_path_labels):
        full_path = os.path.join(folder_path_labels, file_name)
        with open(full_path, 'r') as label_file:
            for line in label_file:
                parts = line.split()
                image = Image.open(os.path.join(folder_path_images, file_name[:-4] + '.jpg'))

                center_x, center_y, width, height = map(float, parts[1:])

                image_width, image_height = image.size

                x1 = (center_x - width / 2) * image_width
                y1 = (center_y - height / 2) * image_height
                x2 = (center_x + width / 2) * image_width
                y2 = (center_y + height / 2) * image_height

                coordinates = (x1, y1, x2, y2)
                cropped_image = image.crop(coordinates)
                cropped_image_np = np.array(cropped_image)
                print("------------")
                data_extracted = extract_features(cropped_image_np)
                print(data_extracted)
                feature_values = list(data_extracted)
                feature_values.insert(0, parts[0])


                writer.writerow(feature_values)

print("Data written to CSV successfully.")
