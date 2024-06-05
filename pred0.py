import os
from IPython import display
import ultralytics
from ultralytics import YOLO
from IPython.display import display, Image
import subprocess
from PIL import Image
import glob 
import re
from shapely.geometry import Polygon, box

# print(ultralytics.checks())

# model0 = YOLO('/home/eshan/Eshan/Study/TY/EDI/sem6Update/ML/DATASET_W_w.pt')

def parse_poly(parts):
    parts = list(map(float, parts.strip().split()))
    label = int(parts[0])
    points = parts[1:]
    vertices = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
    return label, vertices

def calculate_polygon_area(vertices):
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - y1 * x2
    print(f'Polygon Area (Road) {abs(area) / 2.0}')
    return abs(area) / 2.0


# Function to convert normalized coordinates to actual pixel coordinates
def parse_bbox(parts):
    label = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])
    
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    
    return label, x_min, y_min, x_max, y_max, width, height

# Function to calculate the area of intersection between two bounding boxes
def calculate_intersection_area(bbox1, bbox2):
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2
    
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)
    
    if x_min_inter < x_max_inter and y_min_inter < y_max_inter:
        inter_width = x_max_inter - x_min_inter
        inter_height = y_max_inter - y_min_inter
        return inter_width * inter_height
    return 0

def custom_sort_key(filename):
    # Extract the number from the filename using regex
    match = re.search(r'images(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return float('inf')  # Put filenames without a number at the end

def predict(images):
    model0 = YOLO('/home/eshan/Eshan/Study/TY/EDI/sem6Update/ML/DATASET_W_w.pt')
    # results = model0(images)
    # for res in images:
    subprocess.run(['yolo','predict',f'model={model0}',f'source={images}','save=True','save_txt=True','show=True'])
    # im = Image.open('ML/runs/detect/predict/val2.jpg')
    # im.show()
        # boxes=res.boxes
        # mask=s=res.masks
        # keypoints=res.keypoints
        # probs=res.probs
        # res.show()
        # res.save(filename=f'{res}.jpg')

def predict1(model, images):
    model1 = YOLO(model)
    res = model1.predict(f'{images}', save=True, save_txt=True, project='pred', name='RoadImages')
    cn = []
    roadArea = 0
    pts = []
    for r in res:
        print(f'boxes: {r.boxes}')
        print(f'msks: {r.masks}')
        print(f'keypoints: {r.keypoints}')
        print(f'probs: {r.probs}')
        print(f'path: {r.path}')
        print(f'classNames: {r.names}')
        cn.append(r.names)
        # r.save(filename=f'runs/detect/predict/{r}.jpg')
    folders = glob.glob('pred/RoadImages*')
    sortF = list(sorted(folders, key=custom_sort_key))
    print(sortF[-2])
    labelsDir = os.path.join(sortF[-2], 'labels')
    txtFs = [f for f in os.listdir(labelsDir) if f.endswith('.txt')]
    if txtFs: 
        txtF = os.path.join(labelsDir, txtFs[0])
        polygons = []
        with open(txtF, 'r') as file:
            lines = file.readlines()
            bboxes = []
            # roadbbox = []
            for line in lines:
                label, vertices = parse_poly(line)
                polygons.append((label, vertices))
            for i,(label, vertices) in enumerate(polygons):
                roadArea = calculate_polygon_area(vertices)
                pts = vertices
    print(f'Poly road Area: {roadArea}')
    return roadArea, pts


def predict0(model,images, roadArea, pts):
    print(f"Road Area Obtained: {roadArea}")
    road_polygon = Polygon(pts)
    model0 = YOLO(model)
    res = model0.predict(f'{images}', save=True, save_txt=True, project='pred', name='images')
    cn = []
    for r in res:
        print(f'boxes: {r.boxes}')
        print(f'msks: {r.masks}')
        print(f'keypoints: {r.keypoints}')
        print(f'probs: {r.probs}')
        print(f'path: {r.path}')
        print(f'classNames: {r.names}')
        cn.append(r.names)
        # r.save(filename=f'runs/detect/predict/{r}.jpg')
    folders = glob.glob('pred/images*')
    sortF = list(sorted(folders, key=custom_sort_key))
    print(sortF[-2])
    labelsDir = os.path.join(sortF[-2], 'labels')
    txtFs = [f for f in os.listdir(labelsDir) if f.endswith('.txt')]
    if txtFs: 
        txtF = os.path.join(labelsDir, txtFs[0])
        # roadArea = 0
        # potholeArea = 0
        # vegArea = 0
        # debArea = 0
        # sigArea = 0
        areas = [0]*4
        # areas.append(roadArea,potholeArea,vegArea,debArea,)

        with open(txtF, 'r') as file:
            lines = file.readlines()
            bboxes = []
            for i,line in enumerate(lines):
                parts = line.split()
                # label = int(parts[0])
                # x_center = float(parts[1])
                # y_center = float(parts[2])
                # width = float(parts[3])
                # height = float(parts[4])
                # if label == 0:
                #     potholeArea += width*height
                # elif label == 1:
                #     vegArea += width*height
                # elif label == 2:
                #     debArea += width*height
                # elif label == 3:
                #     sigArea += width*height
                label, x_min, y_min, x_max, y_max, width, height = parse_bbox(parts)
                bboxes.append((label, x_min, y_min, x_max, y_max, width, height))
            for i, (label, x_min, y_min, x_max, y_max, width, height) in enumerate(bboxes):
                area = width * height
                overlap_area = 0
                for j in range(i):
                    _, x_min2, y_min2, x_max2, y_max2, _, _ = bboxes[j]
                    overlap_area += calculate_intersection_area((x_min, y_min, x_max, y_max), (x_min2, y_min2, x_max2, y_max2))
                net_area = area - overlap_area
                bbox_polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
                if road_polygon.intersects(bbox_polygon):
                    intersection_area = road_polygon.intersection(bbox_polygon).area
                    net_area = min(net_area, intersection_area)
                    print(f"Label: {label} | Areas: {net_area}")
                    areas[label] += net_area
                # net_area = area - overlap_area
                print(f"Label: {label} | Areas : {area}")
                areas[label] += net_area

        # Define the scoring constants
    k1 = 1.0  # Positive impact for road
    k2 = 10.0  # Negative impact for potholes
    k3 = 10.0  # Negative impact for vegetation
    k4 = 10.0  # Negative impact for debris
    k5 = -10  # Positive impact for signage

    # Calculate the road quality score
    score =  k1 * roadArea - k2 * areas[0] - k3 * areas[1] - k4 * areas[2] + k5 * areas[3]
    score = 100 - (((k2*areas[0] + k3*areas[1] + k4*areas[2] + k5*areas[3])/roadArea)*100)
    # print(folders[-1][-1])
    print(f" Road Area: {roadArea} | Potholes Area: {areas[0]} | Vegetation Area: {areas[1]}| Debris Area: {areas[2]} | ")
    print(f"Road quality score: {score}")
    areas.append(roadArea)
    return cn, r.path, score, areas

# def predict0(model,images, roadArea, pts):
#     model0 = YOLO(model)
#     res = model0.predict(f'{images}', save=True, save_txt=True, project='pred', name='images')
#     cn = []
#     for r in res:
#         print(f'boxes: {r.boxes}')
#         print(f'msks: {r.masks}')
#         print(f'keypoints: {r.keypoints}')
#         print(f'probs: {r.probs}')
#         print(f'path: {r.path}')
#         print(f'classNames: {r.names}')
#         cn.append(r.names)
#         # r.save(filename=f'runs/detect/predict/{r}.jpg')
#     folders = glob.glob('pred/images*')
#     sortF = list(sorted(folders, key=custom_sort_key))
#     print(sortF[-2])
#     labelsDir = os.path.join(sortF[-2], 'labels')
#     txtFs = [f for f in os.listdir(labelsDir) if f.endswith('.txt')]
#     if txtFs: 
#         txtF = os.path.join(labelsDir, txtFs[0])
#         areas = [0]*3
#         polygons = []
#         # areas.append(roadArea,potholeArea,vegArea,debArea,)

#         with open(txtF, 'r') as file:
#             lines = file.readlines()
#             bboxes = []
#             # roadbbox = []
#             for line in lines:
#                 label, vertices = parse_poly(line)
#                 polygons.append((label, vertices))
#             for i,(label, vertices) in enumerate(polygons):
#                 area = calculate_polygon_area(vertices)
#                 areas[label] = area
#             for i, (label, vertices) in enumerate(otherbbox):
#                 for j in range(i):
#                     _, other_vertices = polygons[j]
#                     overlap_area += calculate_polygon_area(vertices)
#                 # net_area = area[label] - overlap_area
#                 area[label] -= overlap_area
#                 print(f"Label: {label} | Areas : {areas}")

#         # Define the scoring constants
#     k1 = 1.0  # Positive impact for road
#     k2 = 2.0  # Negative impact for potholes
#     k3 = 1.5  # Negative impact for vegetation
#     k4 = 2.0  # Negative impact for debris
#     k5 = 0.5  # Positive impact for signage

#     # Calculate the road quality score
#     score =  k1 * areas[0] - k2 * areas[1] - k3 * areas[2] - k4 * areas[3] + k5 * areas[4]
#     # print(folders[-1][-1])
#     print(f"Potholes Area: {areas[1]} | Vegetation Area: {areas[2]} | Debris Area: {areas[3]} | Signage Area: {areas[4]}")
#     print(f"Road quality score: {score}")
#     return cn, r.path, score, areas

    # r.save_txt()
# predict0('ML/images/val2.jpg')

folders = glob.glob('pred/images*')
# print(folders)
# print(sorted(folders))
# Sort the filenames using the custom key
sorted_filenames = list(sorted(folders, key=custom_sort_key))
# print(sorted_filenames[-2])

# print(list(folders[-1])[-1])
# im ='ML/images/val2.jpg'

# print(im.split('/')[-1])

# predict(['ML/images/val2.jpg'])

 # Freeze
# freeze = 10 #backbone layers
# freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
# for k, v in model0.model.named_parameters():
#     v.requires_grad = True  # train all layers
#     if any(x in k for x in freeze):
#         print(f'freezing {k}')
#         v.requires_grad = False

# subprocess.run(['yolo','predict','model=/home/eshan/Eshan/Study/TY/EDI/sem6Update/ML/DATASET_W_w.pt','source=/home/eshan/Eshan/Study/TY/EDI/val2.jpg','save=True','save_txt=True','show=True'])
