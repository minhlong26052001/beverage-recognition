from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import os
import numpy as np
import tensorflow as tf
import cv2

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


def load_model(model_dir):
    return tf.saved_model.load(model_dir)


def load_labels(label_map_path):
    return label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)


def load_image_paths(images_path):
    return os.listdir(images_path)


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


def show_inference(model, labels, image_path):
    image_np = np.array(Image.open(image_path))
    output_dict = run_inference_for_single_image(model, image_np)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        labels,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    cv2.imshow(image_path, image_np)
    cv2.waitKey()


def main(verbose=True):
    model_dir = 'my_model/saved_model'
    label_map_path = 'assets/dataset/annotations/labelmap.pbtxt'
    images_dir = 'assets/dataset/validation'

    model = load_model(model_dir)
    if verbose:
        print('Model is loaded.')
        print(model.signatures['serving_default'].inputs)
        print(model.signatures['serving_default'].output_dtypes)
        print(model.signatures['serving_default'].output_shapes)

    labels = load_labels(label_map_path)
    if verbose:
        print('Labels =', labels)

    image_paths = load_image_paths(images_dir)
    if verbose:
        print('Loaded', len(image_paths), 'images.')

    img_dir = os.path.join(os.getcwd(), images_dir)
    for image_path in image_paths:
        show_inference(model, labels, os.path.join(img_dir, image_path))


main(verbose=True)
