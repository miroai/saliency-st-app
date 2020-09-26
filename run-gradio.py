import os
from main import main
import random
import imageio
from PIL import Image
import gradio as gr
import numpy as np


def best_window(saliency, aspect_ratio=(16,9)):
  """
  saliency is np.array with shape (height, width)
  aspect_ratio is tuple of (width, height)
  """
  orig_width, orig_height = saliency.shape
  move_vertically = orig_height >= orig_width / aspect_ratio[0] * aspect_ratio[1]
  if move_vertically:
    saliency_per_row = np.sum(saliency, axis=1)
    height = round(saliency.shape[1] / aspect_ratio[0] * aspect_ratio[1])
    convolved_saliency = np.convolve(saliency_per_row, np.ones(height), "valid")
    max_row = np.argmax(convolved_saliency)
    return 0, orig_width, max_row, max_row + height
  else:
    saliency_per_col = np.sum(saliency, axis=0)
    width = round(saliency.shape[0] / aspect_ratio[1] * aspect_ratio[0])
    convolved_saliency = np.convolve(saliency_per_col, np.ones(width), "valid")
    max_col = np.argmax(convolved_saliency)
    return max_col, max_col + width, 0, orig_height

def predict(img):
  tmp_name = str(random.getrandbits(32))
  tmp_file = 'tmp/{}.jpg'.format(tmp_name)
  imageio.imwrite(tmp_file, img)
  main(tmp_file)
  tmp_result = 'results/images/{}.jpeg'.format(tmp_name)
  map = np.array(Image.open(tmp_result))
  os.remove(tmp_file)
  os.remove(tmp_result)
  left, right, bottom, top = best_window(map)
  out = img[bottom:top, left:right, :]
  return out

thumbnail = "https://ibb.co/y8nh3Mj"
gr.Interface(predict, "image", "image", title="Twitter Image Cropper",
             description="A model similar to Twitter's Image Cropper",
             thumbnail=thumbnail).launch()
