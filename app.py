import streamlit as st

import os, sys, io
import urllib.request as urllib
import numpy as np
from PIL import Image

from run_gradio import load_model, test_model

### Some Utils Functions ###
def get_image(st_asset = st.sidebar, as_np_arr = False, extension_list = ['jpg', 'jpeg', 'png']):
	image_url, image_fh = None, None
	if st_asset.checkbox('use image URL?'):
		image_url = st_asset.text_input("Enter Image URL")
	else:
		image_fh = st_asset.file_uploader(label = "Update your image", type = extension_list)

	im = None
	if image_url:
		response = urllib.urlopen(image_url)
		im = Image.open(io.BytesIO(bytearray(response.read())))
	elif image_fh:
		im = Image.open(image_fh)

	if im and as_np_arr:
		im = np.array(im)
	return im

def show_miro_logo(use_column_width = False, width = 100, st_asset= st.sidebar):
	logo_url = 'https://miro.medium.com/max/1400/0*qLL-32srlq6Y_iTm.png'
	st_asset.image(logo_url, use_column_width = use_column_width, channels = 'BGR', output_format = 'PNG', width = width)

def im_draw_bbox(pil_im, x0, y0, x1, y1, color = 'black', width = 3, caption = None,
			bbv_label_only = False):
	'''
	draw bounding box on the input image pil_im in-place
	Args:
		color: color name as read by Pillow.ImageColor
		use_bbv: use bbox_visualizer
	'''
	import bbox_visualizer as bbv
	if any([type(i)== float for i in [x0,y0,x1,y1]]):
		warnings.warn(f'im_draw_bbox: at least one of x0,y0,x1,y1 is of the type float and is converted to int.')
		x0 = int(x0)
		y0 = int(y0)
		x1 = int(x1)
		y1 = int(y1)

	if bbv_label_only:
		if caption:
			im_array = bbv.draw_flag_with_label(np.array(pil_im),
						label = caption,
						bbox = [x0,y0,x1,y1],
						line_color = ImageColor.getrgb(color),
						text_bg_color = ImageColor.getrgb(color)
						)
		else:
			raise ValueError(f'im_draw_bbox: bbv_label_only is True but caption is None')
	else:
		im_array = bbv.draw_rectangle(np.array(pil_im),
					bbox = [x0, y0, x1, y1],
					bbox_color = ImageColor.getrgb(color),
					thickness = width
					)
		im_array = bbv.add_label(
					im_array, label = caption,
					bbox = [x0,y0,x1,y1],
					text_bg_color = ImageColor.getrgb(color)
					)if caption else im_array
	return Image.fromarray(im_array)

### Streamlit App ###

def Main(model_dict):
	st.set_page_config(layout = 'wide')
	show_miro_logo()
	with st.sidebar.expander('Saliency Demo'):
		st.info('todo: add details about the app')

	im = get_image(st_asset = st.sidebar.expander('Input Image', expanded = True), extension_list = ['jpg','jpeg'])
	if im:
		saliency_im = test_model(np.array(im), model_dict = model_dict)

		l_col, r_col = st.columns(2)
		l_col.image(im, caption = 'Input Image')
		r_col.image(saliency_im, caption = 'Saliency Map')
	else:
		st.warning(f':point_left: please provide an image')

if __name__ == '__main__':
	model_dict = load_model()
	Main(model_dict = model_dict)
