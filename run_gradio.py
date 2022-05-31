from PIL import Image, ImageDraw
import gradio as gr
import numpy as np
import tensorflow as tf
import download, os, sys


def best_window(saliency, aspect_ratio=(16, 9)):
    """
  saliency is np.array with shape (height, width)
  aspect_ratio is tuple of (width, height)
  """
    orig_height, orig_width = saliency.shape
    move_vertically = orig_height >= orig_width / aspect_ratio[0] * \
                      aspect_ratio[1]
    if move_vertically:
        saliency_per_row = np.sum(saliency, axis=1)
        height = round(orig_width / aspect_ratio[0] * aspect_ratio[1])
        convolved_saliency = np.convolve(saliency_per_row, np.ones(height),
                                         "valid")
        max_row = np.argmax(convolved_saliency)
        return 0, orig_width, max_row, max_row + height
    else:
        saliency_per_col = np.sum(saliency, axis=0)
        width = round(orig_height / aspect_ratio[1] * aspect_ratio[0])
        convolved_saliency = np.convolve(saliency_per_col, np.ones(width),
                                         "valid")
        max_col = np.argmax(convolved_saliency)
        return max_col, max_col + width, 0, orig_height


def overlay_saliency(img, map, bbox = {}):
    background = img.convert("RGBA")
    overlay = map.convert("RGBA")
    overlaid = Image.blend(background, overlay, 0.75)
    draw = ImageDraw.Draw(overlaid)
    if bbox:
        draw.rectangle(
            [bbox['left'], bbox['bottom'], bbox['right'], bbox['top']],
            outline="orange", width=5)
    return overlaid


def get_saliency_sum_box(crop_data, bounded, saliency):
    left, right, bottom, top = int(crop_data["x"]), int(
        crop_data["x"] + crop_data["width"]), int(crop_data["y"]), int(
        crop_data["y"] + crop_data["height"])
    sal_sum = np.sum(saliency[bottom:top, left:right])
    total = np.sum(saliency)
    pct_sal = round(100 * sal_sum / total, 2)
    draw = ImageDraw.Draw(bounded)
    draw.rectangle([left, bottom, right, top], outline="red", width=5)
    return bounded, pct_sal


def test_model(im_arr, model_dict):
    # original_arr, crop_data = original_arr
    # crop_data["original_height"] = original_arr.shape[0]
    # crop_data["original_width"] = original_arr.shape[1]
    original_img = Image.fromarray(im_arr).convert('RGB')
    w, h = original_img.size
    h_ = int(400 / w * h)
    resized_img = original_img.resize((400, h_))
    resized_arr = np.asarray(resized_img)

    resized_arr = resized_arr[np.newaxis, ...]
    saliency_arr = model_dict['sess'].run(model_dict['predicted_maps'],
                                feed_dict={
                                    model_dict['input_plhd']: resized_arr
                                })
    saliency_arr = saliency_arr.squeeze()

    saliency_img = Image.fromarray(np.uint8(saliency_arr * 255), 'L')
    saliency_resized_img = saliency_img.resize((w, h))

    saliency_resized_arr = np.asarray(saliency_resized_img)
    saliency_zero_one = np.divide(saliency_resized_arr, 255.0)
    # left, right, bottom, top = best_window(saliency_resized_arr)
    # output = original_arr[bottom:top, left:right, :]

    # bounded = overlay_saliency(original_img, saliency_resized_img,
                               # left, right, bottom, top)
    bounded = overlay_saliency(original_img, saliency_resized_img)
    return bounded
    # with_sal_box, pct_sal = get_saliency_sum_box(crop_data, bounded,
    #                                              saliency_zero_one)
    # sal_sum = str(pct_sal) + "%"
    # return with_sal_box, sal_sum

def load_model(model_name = "weights/model_mit1003_cpu.pb"):
    ### Model loading code
    graph_def = tf.GraphDef()
    if not os.path.isfile(model_name):
        download.download_pretrained_weights('weights/', 'model_mit1003_cpu')

    with tf.gfile.Open(model_name, "rb") as file:
        graph_def.ParseFromString(file.read())
        input_plhd = tf.placeholder(tf.float32, (None, None, None, 3))
        [predicted_maps] = tf.import_graph_def(graph_def,
                                               input_map={"input": input_plhd},
                                               return_elements=["output:0"])

    sess = tf.Session()
    return {
        'sess': sess,
        'predicted_maps': predicted_maps,
        'input_plhd': input_plhd
    }

if __name__ == '__main__':
    examples = [["images/1.jpg", True],
                ["images/2.jpg", True]]

    thumbnail = "https://ibb.co/hXdbDyD"
    io = gr.Interface(test_model,
                      gr.inputs.Image(label="Your Image", tool='select'),
                      [gr.outputs.Image(label="Cropped Image"),
                       gr.outputs.Label(label="Percent of Saliency in Red Box")],
                      allow_flagging=False,
                      thumbnail=thumbnail,
                      examples=examples, analytics_enabled=False)

    io.launch(debug=True)
