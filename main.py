import json
import cv2
import numpy as np
from PIL import Image
import csv
import matplotlib.pyplot as plt


def load_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def read_image(image_path):
    with Image.open(image_path) as img:
        image = np.array(img)
    return image


def generate_masks_and_calculate_areas(data, image, label_names):
    areas = {label: 0 for label in label_names}
    image_size = image.shape[:2]
    inside_mask = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    for label_name in label_names:
        for shape in data['shapes']:
            if shape['label'] == label_name:
                label_points = np.array(shape['points']).round().astype(
                    int).astype(np.int32)
                cv2.fillPoly(inside_mask, pts=[
                             label_points], color=(255, 255, 255))
        inside_mask_single_channel = cv2.cvtColor(
            inside_mask, cv2.COLOR_BGR2GRAY)
        areas[label_name] = cv2.countNonZero(inside_mask_single_channel)
    return areas


def save_areas_to_csv(areas, csv_file_path):
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Label Name", "Area"])
        for label, area in areas.items():
            writer.writerow([label, area])


def read_areas_from_csv(csv_file_path):
    areas = {}
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            areas[row[0]] = int(row[1])
    return areas


def calculate_percentages(areas):
    total_area = sum(areas.values())
    percentages = {label: (area / total_area) *
                   100 for label, area in areas.items()}
    return percentages


def save_percentages_to_csv(percentages, csv_file_path):
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Label Name", "Percentage"])
        for label, percentage in percentages.items():
            writer.writerow([label, percentage])


def create_pie_chart(percentages, title, file_path):
    labels = list(percentages.keys())
    sizes = list(percentages.values())
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title(title)
    plt.axis('equal')
    plt.savefig(file_path)
    plt.show()


def process_image_and_json(json_file_path, image_path):
    data = load_json(json_file_path)
    image = read_image(image_path)
    label_names = ['RA Appendage', 'RV Myocardium',
                   'Aorta', 'RV & PA Epicardial Fat']
    areas = generate_masks_and_calculate_areas(data, image, label_names)
    areas_csv = 'areas.csv'
    save_areas_to_csv(areas, areas_csv)
    percentages = calculate_percentages(areas)
    percentages_csv = 'percentages.csv'
    save_percentages_to_csv(percentages, percentages_csv)
    create_pie_chart(percentages, 'Heart Area Breakdown',
                     'heart_area_breakdown.png')


if __name__ == '__main__':
    json_file_path = 'pt_3.json'
    image_path = 'pt_3.png'
    process_image_and_json(json_file_path, image_path)
