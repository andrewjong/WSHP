"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import sparse
from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels
from tqdm import trange

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

NUM_CLASSES = 7
DATA_LIST = "./dataset/dance.txt"
SAVE_DIR = "./output/"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("img_path", type=str, help="Path to the RGB image file folder.")
    parser.add_argument(
        "model_weights", type=str, help="Path to the file with model weights."
    )
    parser.add_argument(
        "--data_list", type=str, default=DATA_LIST, help="Path to the image list."
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help="Number of classes to predict (including background).",
    )
    parser.add_argument(
        "--save-dir", type=str, default=SAVE_DIR, help="Where to save predicted mask."
    )
    return parser.parse_args()


def load(saver, sess, ckpt_path):
    """Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    """
    saver.restore(sess, ckpt_path)
    print(("Restored model parameters from {}".format(ckpt_path)))


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    num_steps = file_len(args.data_list)
    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.img_path,
            args.data_list,
            None,  # No defined input size.
            False,  # No random scale.
            False,  # No random mirror.
            255,
            IMG_MEAN,
            coord,
        )
        image, label = reader.image, reader.label
        title = reader.queue[0]
    image_batch, label_batch = (
        tf.expand_dims(image, axis=0),
        tf.expand_dims(label, axis=0),
    )  # Add one batch dimension.

    # Create network.
    net = DeepLabResNetModel(
        {"data": image_batch}, is_training=False, num_classes=args.num_classes
    )

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers["fc1_voc12"]
    before_argmax = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(before_argmax, dimension=3)
    pred = tf.expand_dims(raw_output_up, axis=3)
    hw_only = pred[0, :, :, 0]
    class_0 = tf.where(tf.equal(hw_only, 0))
    class_1 = tf.where(tf.equal(hw_only, 1))
    class_2 = tf.where(tf.equal(hw_only, 2))
    class_3 = tf.where(tf.equal(hw_only, 3))
    class_4 = tf.where(tf.equal(hw_only, 4))
    class_5 = tf.where(tf.equal(hw_only, 5))
    class_6 = tf.where(tf.equal(hw_only, 6))

    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.model_weights)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    start_time = time.time()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # write the header
    rois_file = os.path.join(args.save_dir, "rois.csv")
    with open(rois_file, "w") as f:
        f.write("id,x1,y1,x2,y2\n")
    # Perform inference.
    t = trange(num_steps, desc="Inference progress", unit="img")
    for step in t:
        # run through the model
        jpg_path, c0, c1, c2, c3, c4, c5, c6, raw_output_up_ = sess.run(
            [
                title,
                class_0,
                class_1,
                class_2,
                class_3,
                class_4,
                class_5,
                class_6,
                raw_output_up,
            ]
        )

        # == First, save the body segmentation ==
        # convert to a compressed matrix, because we have a lot of 0's for the
        # background
        compressed = sparse.csr_matrix(np.squeeze(raw_output_up_))
        fname = os.path.splitext(os.path.basename(str(jpg_path)))[0]
        out = os.path.join(args.save_dir, "body", fname)
        sparse.save_npz(out, compressed)

        # == Next, save the ROIs ==

        for c in (c0, c1, c2, c3, c4, c5, c6):
            try:
                min_x = np.min(c[:, 1])
            except ValueError:
                min_x = None
            try:
                min_y = np.min(c[:, 0])
            except ValueError:
                min_y = None
            try:
                max_x = np.max(c[:, 1])
            except ValueError:
                max_x = None
            try:
                max_y = np.max(c[:, 0])
            except ValueError:
                max_y = None
            # write out the stuff
            with open(rois_file, "a") as f:
                f.write(
                    ",".join(
                        (str(step), str(min_x), str(min_y), str(max_x), str(max_y),
                         "\n")
                    )
                )

        # msk = decode_labels(preds, num_classes=args.num_classes)
        # # the mask
        # im = Image.fromarray(msk[0])
        # # save the mask
        # jpg_path = str(jpg_path).split('/')[-1].split('.')[0]
        # out = os.path.join(args.save_dir, jpg_path + '.png')
        # im.save(out)
        t.set_description("Finished " + fname)

        # AJ: We want to save only the mask, not the background. therefore we commented out below

        # # the original image
        # img_o = Image.open(jpg_path)
        # jpg_path = jpg_path.split('/')[-1].split('.')[0]
        # # create the final result using the mask and the original
        # img = np.array(im)*0.9 + np.array(img_o)*0.7
        # # clip surpassed colors
        # img[img>255] = 255
        # img = Image.fromarray(np.uint8(img))
        # img.save(args.save_dir + jpg_path + '.png')
        # # print('Image processed {}.png'.format(jpg_path))

    total_time = time.time() - start_time
    print(("The output files have been saved to {}".format(args.save_dir)))
    print(("It took {} sec on each image.".format(total_time / num_steps)))


if __name__ == "__main__":
    main()
