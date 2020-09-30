import argparse
import os

import tensorflow as tf

import config
import data
import download

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
    

def better_test_model(dataset, paths, device):
    """The main function for executing network testing. It loads the specified
       dataset iterator and optimized saliency model. By default, when no model
       checkpoint is found locally, the pretrained weights will be downloaded.
       Testing only works for models trained on the same device as specified in
       the config file.

    Args:
        dataset (str): Denotes the dataset that was used during training.
        paths (dict, str): A dictionary with all path elements.
        device (str): Represents either "cpu" or "gpu".
    """
    jpeg = data.postprocess_saliency_map(predicted_maps[0],
                                         original_shape[0])

    print(">> Start testing with %s %s model..." % (dataset.upper(), device))

    with tf.Session() as sess:
        sess.run(init_op)

        while True:
            try:
                output_file, path = sess.run([jpeg, file_path])
            except tf.errors.OutOfRangeError:
                break

            path = path[0][0].decode("utf-8")

            filename = os.path.basename(path)
            filename = os.path.splitext(filename)[0]
            filename += ".jpeg"

            os.makedirs(paths["images"], exist_ok=True)

            with open(paths["images"] + filename, "wb") as file:
                file.write(output_file)

                
def test_model(dataset, paths, device):
    """The main function for executing network testing. It loads the specified
       dataset iterator and optimized saliency model. By default, when no model
       checkpoint is found locally, the pretrained weights will be downloaded.
       Testing only works for models trained on the same device as specified in
       the config file.

    Args:
        dataset (str): Denotes the dataset that was used during training.
        paths (dict, str): A dictionary with all path elements.
        device (str): Represents either "cpu" or "gpu".
    """

    iterator = data.get_dataset_iterator("test", dataset, paths["data"])

    next_element, init_op = iterator

    input_images, original_shape, file_path = next_element

    graph_def = tf.GraphDef()

    model_name = "model_%s_%s.pb" % (dataset, device)

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

    jpeg = data.postprocess_saliency_map(predicted_maps[0],
                                         original_shape[0])

    print(">> Start testing with %s %s model..." % (dataset.upper(), device))

    tf.reset_default_graph()
    
    with tf.Session() as sess:
        sess.run(init_op)

        while True:
            try:
                output_file, path = sess.run([jpeg, file_path])
            except tf.errors.OutOfRangeError:
                break

            path = path[0][0].decode("utf-8")

            filename = os.path.basename(path)
            filename = os.path.splitext(filename)[0]
            filename += ".jpeg"

            os.makedirs(paths["images"], exist_ok=True)

            with open(paths["images"] + filename, "wb") as file:
                file.write(output_file)


def main(tmp_path):
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    current_path = os.path.dirname(os.path.realpath(__file__))

    args = argparse.Namespace(path='{}'.format(tmp_path))

    paths = define_paths(current_path, args)

    test_model('mit1003', paths, config.PARAMS["device"])

