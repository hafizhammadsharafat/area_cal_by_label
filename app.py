from flask import Flask, request, jsonify, send_file
from flask import Flask, request, jsonify, send_file, send_from_directory
import os
import json
import cv2
import numpy as np
from PIL import Image
import csv
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

STATIC_DIR = os.path.abspath("static")


@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


def load_json(json_file):
    data = json.loads(json_file.read())
    return data


def read_image(image_file):
    image = np.array(Image.open(image_file))
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


def calculate_percentages(areas):
    total_area = sum(areas.values())
    percentages = {label: (area / total_area) *
                   100 for label, area in areas.items()}
    return percentages


def create_pie_chart(percentages):
    labels = list(percentages.keys())
    sizes = list(percentages.values())
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('Heart Area Breakdown')
    plt.axis('equal')
    chart_file = 'pie_chart.png'
    plt.savefig(chart_file, format='png')
    plt.close()
    return chart_file


@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files or 'json' not in request.files:
        return jsonify({'error': 'Image and JSON files are required.'}), 400

    image_file = request.files['image']
    json_file = request.files['json']

    if image_file.filename == '' or json_file.filename == '':
        return jsonify({'error': 'Image and JSON files are required.'}), 400

    if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid image format. Supported formats are PNG, JPG, and JPEG.'}), 400

    if not json_file.filename.lower().endswith('.json'):
        return jsonify({'error': 'Invalid JSON file format. Must be a JSON file.'}), 400

    try:
        data = load_json(json_file)
        image = read_image(image_file)
        label_names = ['RA Appendage', 'RV Myocardium',
                       'Aorta', 'RV & PA Epicardial Fat']
        areas = generate_masks_and_calculate_areas(data, image, label_names)
        percentages = calculate_percentages(areas)
        chart_file = create_pie_chart(percentages)
        return send_file(chart_file, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
