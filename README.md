# Road Quality Assessment

## Table of Contents
- Introduction
- Features
- Setup
- Preprocessing
- Training
- Results

## Introduction

This project uses a YOLOv8 model to assess road quality by detecting various faults such as cracks, potholes, vegetation, debris, and signage. The model is fine-tuned on a custom dataset to perform instance segmentation with polygon boxes, ensuring 
accurate fault detection within the road area. The project is deployed on Streamlit, allowing users to upload images and receive annotated images with bounding boxes and a calculated road quality score.

## Features

- Instance Segmentation: Detects and segments objects within the road area using polygon boxes.
- Fault Detection: Identifies and classifies potholes, vegetation, debris, and signage.
- Road Quality Score: Calculates a road quality score based on detected faults within the road area.
- Streamlit Deployment: Provides a user-friendly web interface for uploading images and viewing results.

## Setup
To run the project: 

1. Clone the repository: `git clone https://github.com/yourusername/road-quality-assessment.git`  `cd road-quality-assessment`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`
4. Usage :
  - Select media type ["Image"/"Video"]
  - Upload a road media.
  - Choose the model you want to use (Model1 or InstanceModel).
  - Click "Run Model" to process the media
  - The Uploaded media with instance segmentation boxes and a road quality score will be displayed

## Methodology
The model is built by finetuning the yolo v8 model. The model is finetuned to better identify and also perform instance segmentation(Creates boxes that completely cover the object) on media featuring vegetation, signage , cracks , potholes and debris 
found on roads. The dataset was prepared manually by creating polygons as class labels for the models. The model classifies vegetaion , debris , signage , potholes as well as the road and outputs a polygon for each. The shapes for the classes are 
re-adjusted as only the area that occurs within the road area are considered. The area of these shapes is calculated using the shoelace formula : `Area, A = 1 2 (x0y1 − x1y0 + ... + xn−2 yn−1 − xn−1 yn−2 + xn−1 y0 − x0yn−1 )` . The overlapping area 
between classes within the road is also removed based in their co-ordinates. Finally the areas are input in the empiracally obtained formula : 
```
k1 = 1.0  # Positive impact for road
k2 = 10.0  # Negative impact for potholes
k3 = 10.0  # Negative impact for vegetation
k4 = 10.0  # Negative impact for debris
k5 = -10  # Positive impact for signage

Calculate the road quality score
score =  k1 * roadArea - k2 * areas[0] - k3 * areas[1] - k4 * areas[2] + k5 * areas[3]
```
