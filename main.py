import streamlit as st
import requests
import io
from PIL import Image
import pandas as pd
import numpy as np
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import tensorflow as tf

st.sidebar.write('#### Select an image to upload.')
uploaded_file = st.sidebar.file_uploader('',
                                         type=['jpg'],
                                         accept_multiple_files=False)

## Add in sliders.
confidence_threshold = st.sidebar.slider(
    'Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?', 0.0, 1.0,
    0.5, 0.01)

## Title.
st.write('# Beverage Recognition Object Detection')

## Pull in default image or user-selected image.
if uploaded_file is None:
    # Default image.
    url = 'https://github.com/KuroShinigami318/beverage-recognition/blob/master/assets/dataset/validation/IMG_20200914_195606.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    # User-selected image.
    image = Image.open(uploaded_file)

## Subtitle.
st.write('### Inferenced Image')


@st.cache
def load_model(model_dir):
    return tf.saved_model.load(model_dir)


@st.cache
def load_labels(label_map_path):
    return label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)


@st.cache
def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


@st.cache
def show_inference(model, labels, img):
    image_np = np.array(img)
    output_dict = run_inference_for_single_image(model, image_np)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        labels,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        min_score_thresh=confidence_threshold,
        line_thickness=8)

    image = Image.fromarray(np.uint8(image_np)).convert('RGB')
    # Convert to JPEG Buffer.
    buffered = io.BytesIO()
    image.save(buffered, quality=90, format='JPEG')
    return image, output_dict


model_dir = 'my_model/saved_model'
label_map_path = 'assets/dataset/annotations/labelmap.pbtxt'

model = load_model(model_dir)
labels = load_labels(label_map_path)
# Display image.
image, output_dict = show_inference(model, labels, image)
st.image(image, use_column_width=True)

# Display chart
st.write('### Beverages')
get_labels = lambda i: labels[i]['name']
detection_labels = []
for i in output_dict['detection_classes']:
    detection_labels.append(get_labels(i))
chart_data = pd.DataFrame(output_dict['detection_scores'], index=detection_labels, columns=['Probability'])
st.bar_chart(chart_data)
