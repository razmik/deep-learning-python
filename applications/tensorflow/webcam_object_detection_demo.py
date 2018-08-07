"""
Models: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
Recent Models: https://research.googleblog.com/2017/11/automl-for-large-scale-image.html?m=1
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import numpy as np
import _thread
from multiprocessing import Queue
from scipy import misc

sys.path.append("models/")
sys.path.append("models/object_detection/")

from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
MODEL_FOLDER = 'models/object_detection/'
# MODEL_NAME = 'faster_rcnn_nas_coco_24_10_2017'
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

input_q = Queue(maxsize=10)
output_q = Queue(maxsize=10)

print('Initialized variables.')


if not os.path.isfile(MODEL_FOLDER+MODEL_FILE):
    print('Downloading model...')
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FOLDER+MODEL_FILE)
else:
    print('Loading model...')

tar_file = tarfile.open(MODEL_FOLDER+MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())
print('Extracting files...')

print('Load a (frozen) Tensorflow model into memory.')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

print('Loading label map')
label_map = label_map_util.load_labelmap(MODEL_FOLDER+PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    im_width, im_height = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def process_frame(image_np, itr):

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            while True:
                image_np = input_q.get()

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                resized_img = misc.imresize(image_np, (1200, 1920, 3), interp='nearest')
                cv2.imshow("Recognizer View", resized_img)
                if cv2.waitKey(1) & 0xFF == ord('w'):
                    break


cap = cv2.VideoCapture(0)
webcam_frame = "Original View"
recog_frame = "Recognizer View"
# cv2.namedWindow(webcam_frame, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(webcam_frame, 400, 320)

iteration = 1
while True:
    ret, frame = cap.read()
    # cv2.imshow(webcam_frame, frame)

    if iteration % 4 == 0:
        input_q.put(frame)

    if iteration == 2:
        _thread.start_new_thread(process_frame, (frame, iteration))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    iteration += 1

cap.release()
cv2.destroyAllWindows()