from pydoc import getpager
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import io
import tensorflow as tf 
import tensorflow.keras as keras
from fashion_detection_model import SSD, Predictor
from utils.utils import draw_boxes, cut_cothes
from color_recognition import get_color, hex2name, rgb2hex
from pattern_recognition import get_pattern
st.set_page_config(page_title='Fashion Recognition',page_icon="👗", layout="wide")


@st.cache
def load_model(model_path_detection, class_names_detection):
    net = SSD(len(class_names_detection), is_test=True)
    net.load(model_path_detection)
    model_detection = Predictor(net, candidate_size=200)
    
    return model_detection


def load_model_pattern(model_path_pattern):
    model_pattern = tf.keras.models.load_model(model_path_pattern)
    return model_pattern


def predict_od(model, img, class_names):
    boxes, labels, probs = model.predict(img, 50, thresh)
    list_clothes = cut_cothes(img, boxes, labels, probs, class_names)
    img_draw = draw_boxes(img, boxes, labels, probs, class_names)

    return img_draw, list_clothes


def plot_color(list_color, values):
    ordered_colors = [list_color[i] for i in values.keys()]
    hex_colors = [rgb2hex(ordered_colors[i]) for i in values.keys()]
    name_color = [hex2name(ordered_colors[i]) for i in values.keys()]
    fig1, ax1 = plt.subplots(figsize=(3,3))
    ax1.pie(values.values(), labels = name_color, colors = hex_colors, startangle=90, textprops={'fontsize': 14})
    ax1.axis('equal')  

    return fig1, ax1

def UI():
    st.markdown("<h1 style='text-align: center; color: black;'>Fashion Recognition</h1>", unsafe_allow_html=True)
    
    _, a, _ = st.columns((1,10, 1))
    a.markdown("<p style='text-align: left; color: black; font-size: 25px;'>A website where you can find all fashion styles in world with only a picture</p>", unsafe_allow_html=True)
    
    img_file_buffer = a.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png','abs'])
    if img_file_buffer is not None:
        img = np.array(Image.open(img_file_buffer))
        img_draw, list_clothes = predict_od(model_od, img, class_names_detection)

        a.image(img_draw)
        a.markdown('---')

        for i in list_clothes:
            _, a, _, b, c,_ = st.columns((0.5,1.5,.3,1.3,1.5, 0.5))
            img_clothes = i['img']
            label_clothes = i['label']
            center_colors, counts = get_color(img_clothes, 3)
            max_color_index = max(counts, key=counts.get)
            main_color = center_colors[max_color_index]
            main_color_name = hex2name(main_color)
            pattern_clothes = get_pattern(img_clothes, model_pattern, class_names_pattern)
            a.image(img_clothes, use_column_width=True)
            b.markdown('## {}'.format(label_clothes))
            b.markdown('#### Pattern: {}'.format(pattern_clothes))
            b.markdown('#### Color: {}'.format(main_color_name))
            c.markdown('### Color')
            fig, ax = plot_color(center_colors, counts)
            c.pyplot(fig)
            _, a, _ = st.columns((1,5, 1))
            a.markdown('---')


if __name__ == "__main__":
    thresh = 0.5
    class_names_detection = ['BACKGROUND', 'Sunglass', 'Hat', 'Jacket', 'Shirt', 'Pants', 'Shorts', 'Skirt', 'Dress', 'Bag', 'Shoe']
    class_names_pattern = ['Floral', 'Plain', 'Dot', 'Squares', 'Stripes']
    model_path_detection = './models/vgg16-ssd-Epoch-125-Loss-2.8042236075681797.pth'
    model_path_pattern = './models/fashion_pattern_41_0.88.h5'
    model_od = load_model(model_path_detection, class_names_detection)
    model_pattern = load_model_pattern(model_path_pattern)
    UI()
