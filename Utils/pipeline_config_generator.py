from xml_to_boxes import xml_to_boxes, kmeans_aspect_ratios
import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2


def get_num_classes(label_map_path):
    label_map = open(label_map_path, 'r')
    items = label_map.read().split('item')[1:]
    return len(items)


checkpoint_path = 'checkpoint/ckpt-0'
label_path = '../assets/dataset/annotations/labelmap.pbtxt'
train_record = 'assets/dataset/annotations/training.record'
test_record = 'assets/dataset/annotations/test.record'
XML_PATH = '../assets/dataset/training/annotations'
num_aspect_ratios = 4  # can be [2,3,4,5,6]
batch_size = 24

# Tune the iterations based on the size and distribution of your dataset
# You can check avg_iou_prec every 100 iterations to see how centroids converge
kmeans_max_iter = 500

# These should match the training pipeline config ('fixed_shape_resizer' param)
width = 320
height = 320

# Get the ground-truth bounding boxes for our dataset
bboxes = xml_to_boxes(path=XML_PATH, rescale_width=width, rescale_height=height)

aspect_ratios, avg_iou_perc = kmeans_aspect_ratios(
    bboxes=bboxes,
    kmeans_max_iter=kmeans_max_iter,
    num_aspect_ratios=num_aspect_ratios)

aspect_ratios = sorted(aspect_ratios)

print('Aspect ratios generated:', [round(ar, 2) for ar in aspect_ratios])
print('Average IOU with anchors:', avg_iou_perc)

pipeline = pipeline_pb2.TrainEvalPipelineConfig()
config_path = '../config/pipeline.config'
with tf.io.gfile.GFile(config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline)

# Set number of classes num_classes.
pipeline.model.ssd.num_classes = get_num_classes(label_path)

while pipeline.model.ssd.anchor_generator.ssd_anchor_generator.aspect_ratios:
    pipeline.model.ssd.anchor_generator.ssd_anchor_generator.aspect_ratios.pop()

for i in range(len(aspect_ratios)):
    pipeline.model.ssd.anchor_generator.ssd_anchor_generator.aspect_ratios.append(aspect_ratios[i])

# checkpoint
pipeline.train_config.fine_tune_checkpoint = checkpoint_path

# set input label_map_path
pipeline.train_input_reader.label_map_path = label_path.replace('../', '')
pipeline.eval_input_reader[0].label_map_path = label_path.replace('../', '')

# set input_path
pipeline.train_input_reader.tf_record_input_reader.input_path[:] = [train_record]
pipeline.eval_input_reader[0].tf_record_input_reader.input_path[:] = [test_record]

# set checkpoint_type
pipeline.train_config.fine_tune_checkpoint_type = 'detection'

# set batch size
pipeline.train_config.batch_size = batch_size

# set data_augmentation_options
pipeline.train_config.data_augmentation_options[1].random_horizontal_flip.keypoint_flip_permutation[:] = [1, 0, 2, 3, 5,
                                                                                                          4]
pipeline.train_config.data_augmentation_options[1].random_horizontal_flip.probability = 0.5

pipeline.train_config.data_augmentation_options[0].random_crop_image.min_object_covered = 0.75
pipeline.train_config.data_augmentation_options[0].random_crop_image.min_aspect_ratio = 0.75
pipeline.train_config.data_augmentation_options[0].random_crop_image.max_aspect_ratio = 1.5
pipeline.train_config.data_augmentation_options[0].random_crop_image.min_area = 0.25
pipeline.train_config.data_augmentation_options[0].random_crop_image.max_area = 0.875
pipeline.train_config.data_augmentation_options[0].random_crop_image.overlap_thresh = 0.5
pipeline.train_config.data_augmentation_options[0].random_crop_image.clip_boxes = False
pipeline.train_config.data_augmentation_options[0].random_crop_image.random_coef = 0.125

config_text = text_format.MessageToString(pipeline)
with tf.io.gfile.GFile(config_path, "wb") as f:
    f.write(config_text)
