import streamlit as st

import os, sys, json
import urllib.request as urllib
import numpy as np
# import pandas as pd
from PIL import Image

### Some Utils Functions ###
def get_image(st_asset = st.sidebar, as_np_arr = False, extension_list = ['jpg', 'jpeg', 'png']):
	image_url = st_asset.text_input("Enter Image URL")
	image_fh = st_asset.file_uploader(label = "Update your image", type = extension_list)

	if image_url and image_fh:
		st_asset.warning(f'image url takes precedence over uploaded image file')

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

def get_pil_im(fp_url_nparray, verbose = False):
	'''
	return a PIL image object
	Args:
		im: np.array, filepath, or url
	'''
    import validators

	im = fp_url_nparray
	pil_im = None
	if type(im) == np.ndarray:
		pil_im = Image.fromarray(im)
	elif os.path.isfile(im):
		pil_im = Image.open(im)
	elif validators.url(im):
		r = urllib.urlopen(im)
		pil_im = Image.open(io.BytesIO(r.read()))
	else:
		raise ValueError(f'get_im: im must be np array, filename, or url')

	if verbose:
		print(f'Find image of size {pil_im.size}')
	return pil_im

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

def Main():
    st.set_page_config(layout = 'wide')
	show_miro_logo()
    with st.sidebar.expander('Saliency Demo'):
        st.info('todo: add details about the app')

    im = get_image(st_asset = st.sidebar.expander('Input Image', expanded = True), extension_list = ['jpg','jpeg'])
    if im:
        st.image(im)
    else:
        st.warning(f':point_left: please provide an image')

if __name__ == '__main__':
	Main()
