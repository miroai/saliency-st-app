import os
from main import main
import random
import imageio
from PIL import Image, ImageDraw
import gradio as gr
import numpy as np
import tensorflow as tf


def best_window(saliency, aspect_ratio=(16,9)):
  """
  saliency is np.array with shape (height, width)
  aspect_ratio is tuple of (width, height)
  """
  orig_height, orig_width = saliency.shape
  move_vertically = orig_height >= orig_width / aspect_ratio[0] * aspect_ratio[1]
  if move_vertically:
    saliency_per_row = np.sum(saliency, axis=1)
    height = round(orig_width / aspect_ratio[0] * aspect_ratio[1])
    convolved_saliency = np.convolve(saliency_per_row, np.ones(height), "valid")
    max_row = np.argmax(convolved_saliency)
    return 0, orig_width, max_row, max_row + height
  else:
    saliency_per_col = np.sum(saliency, axis=0)
    width = round(orig_height / aspect_ratio[1] * aspect_ratio[0])
    convolved_saliency = np.convolve(saliency_per_col, np.ones(width), "valid")
    max_col = np.argmax(convolved_saliency)
    return max_col, max_col + width, 0, orig_height


def overlay_saliency(img, map, left, right, bottom, top):
  background = img.convert("RGBA")
  overlay = map.convert("RGBA")
  overlaid = Image.blend(background, overlay, 0.75)
  draw = ImageDraw.Draw(overlaid)
  draw.rectangle([left, bottom, right, top], outline ="orange", width=5)
  return overlaid

def test_model(original_arr, show_saliency): 
    original_img = Image.fromarray(original_arr).convert('RGB')
    w, h = original_img.size
    h_ = int(400 / w * h)
    resized_img = original_img.resize((400, h_))
    resized_arr = np.asarray(resized_img)

    resized_arr = resized_arr[np.newaxis, ...]
    saliency_arr = sess.run(predicted_maps, feed_dict={input_plhd: resized_arr})
    saliency_arr = saliency_arr.squeeze()

    saliency_img = Image.fromarray(np.uint8(saliency_arr * 255) , 'L')
    saliency_resized_img = saliency_img.resize((w, h))
    saliency_resized_arr = np.asarray(saliency_resized_img)

    left, right, bottom, top = best_window(saliency_resized_arr)
    output = original_arr[bottom:top, left:right, :]
    
    if show_saliency:
       bounded = overlay_saliency(original_img, saliency_resized_img, 
        left, right, bottom, top)
       return bounded

    return output


### Model loading code
graph_def = tf.GraphDef()
model_name = "weights/model_mit1003_cpu.pb"

with tf.gfile.Open(model_name, "rb") as file:
    graph_def.ParseFromString(file.read())
    input_plhd = tf.placeholder(tf.float32, (None, None, None, 3))
    [predicted_maps] = tf.import_graph_def(graph_def,
                                           input_map={"input": input_plhd},
                                           return_elements=["output:0"])

sess = tf.Session()


examples=[["images/1.jpg", True],
          ["images/2.jpg", True]]

thumbnail = "https://ibb.co/hXdbDyD"
io = gr.Interface(test_model, 
  [gr.inputs.Image(label="Your Image"), gr.inputs.Checkbox(label="Show Saliency Map")],
  gr.outputs.Image(label="Cropped Image"), 
  allow_flagging=False, 
  thumbnail=thumbnail, 
  examples=examples)

io.launch()

