import argparse
import os
import tensorflow as tf
import config
import data
import download
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)


def define_paths(current_path, args):
    """A helper function to define all relevant path elements for the
       locations of data, weights, and the results from either training
       or testing a model.

    Args:
        current_path (str): The absolute path string of this script.
        args (object): A namespace object with values from command line.

    Returns:
        dict: A dictionary with all path elements.
    """
    
    if args is None:
        data_path = None
    else:
        if os.path.isfile(args.path):
            data_path = args.path
        else:
            data_path = os.path.join(args.path, "")

    results_path = current_path + "/results/"
    weights_path = current_path + "/weights/"

    history_path = results_path + "history/"
    images_path = results_path + "images/"
    ckpts_path = results_path + "ckpts/"

    best_path = ckpts_path + "best/"
    latest_path = ckpts_path + "latest/"

    # if args.phase == "train":
    #     if args.data not in data_path:
    #         data_path += args.data + "/"

    paths = {
        "data": data_path,
        "history": history_path,
        "images": images_path,
        "best": best_path,
        "latest": latest_path,
        "weights": weights_path
    }

    return paths

def get_tf_objects(paths):
    dataset = 'mit1003'
    device = config.PARAMS["device"]
    model_name = "model_%s_%s.pb" % (dataset, device)
    
    current_path = os.path.dirname(os.path.realpath(__file__))
    paths = define_paths(current_path, None)
    

    if os.path.isfile(paths["best"] + model_name):
        with tf.gfile.Open(paths["best"] + model_name, "rb") as file:
            graph_def.ParseFromString(file.read())
    else:
        if not os.path.isfile(paths["weights"] + model_name):
            download.download_pretrained_weights(paths["weights"],
                                                 model_name[:-3])

        with tf.gfile.Open(paths["weights"] + model_name, "rb") as file:
            graph_def.ParseFromString(file.read())

    [predicted_maps] = tf.import_graph_def(graph_def,
                                           input_map={"input": input_images},
                                           return_elements=["output:0"])
    
    return 
    

                

def main(tmp_path):
    return test_model()
