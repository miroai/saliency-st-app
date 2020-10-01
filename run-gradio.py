import os
from main import main
import random
import imageio
from PIL import Image, ImageDraw
import gradio as gr
import numpy as np


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


def predict(img, show_saliency):
  tmp_name = str(random.getrandbits(64))
  tmp_file = 'tmp/{}.jpg'.format(tmp_name)
  imageio.imwrite(tmp_file, img)
  main(tmp_file)
  tmp_result = 'results/images/{}.jpeg'.format(tmp_name)
  map_pil = Image.open(tmp_result)
  img_pil = Image.open(tmp_file)
  os.remove(tmp_file)
  os.remove(tmp_result)
  map = np.array(map_pil)
  left, right, bottom, top = best_window(map)
  out = img[bottom:top, left:right, :]
  if show_saliency:
     bounded = overlay_saliency(img_pil, map_pil, left, right, bottom, top)
     return bounded
  return out

main('tmp/example.png')

examples=[["images/1.jpg", True],
          ["images/2.jpg", True]]

thumbnail = "https://ibb.co/hXdbDyD"
gr.Interface(predict, [gr.inputs.Image(label="Your Image"),
                       gr.inputs.Checkbox(label="Show Saliency Map")],
             gr.outputs.Image(label="Cropped Image"), allow_flagging=False, thumbnail=thumbnail, examples=examples).launch()

