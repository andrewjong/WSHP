"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

import argparse
import os
import time
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import sparse
from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels
from tqdm import trange

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

NUM_CLASSES = 7

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Body segmentation inference, customized for andrew's work",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "img_path",
        type=str,
        help="Path to the top level data directory, containing a 'texture' subfolder.",
    )
    parser.add_argument(
        "--data_list", default="list.txt", help="Path to the image list relative to img_path"
    )
    parser.add_argument(
        "--body_dir",
        type=str,
        default="body",
        help="Where to save body mask relative to img_path.",
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default="body_vis",
        help="Where to save visualized mask relative to img_path.",
    )
    parser.add_argument(
        "--visualize_step",
        type=int,
        default=1000,
        help="how often in steps to visualize the mask",
    )
    parser.add_argument(
        "--no_body",
        action="store_true",
        help="choose not to output body segmentation compressed npz arrays",
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="choose not to output body segmentation visualizations",
    )
    parser.add_argument(
        "--no_rois",
        action="store_true",
        help="choose not to output rois",
    )
    parser.add_argument(
        "--model_weights",
        default="models/final_model/model.ckpt-19315",
        type=str,
        help="Path to the file with model weights.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=NUM_CLASSES,
        help="Number of classes to predict (including background).",
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


def write_header(rois_file):
    with open(rois_file, "w") as f:
        f.write("id,x1,y1,x2,y2\n")


# AJ function
def extract_nums_only(frame_name):
    as_str = "".join(filter(lambda x: x.isdigit(), frame_name))
    # strip leading 0s
    as_str = as_str.lstrip("0")
    return as_str


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    num_steps = file_len(os.path.join(args.img_path, args.data_list))
    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            os.path.join(args.img_path, "texture"),
            os.path.join(args.img_path, args.data_list),
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
    os.makedirs(os.path.join(args.img_path, args.body_dir), exist_ok=True)
    os.makedirs(os.path.join(args.img_path, args.vis_dir), exist_ok=True)

    # write the header
    rois_file = os.path.join(args.img_path, "rois.csv")
    if os.path.isfile(rois_file):
        print(f"The rois file {rois_file} already exists...")
        ans = None
        while all(ans != choice for choice in ("a", "o", "q")):
            ans = input("Do you want to (a)ppend, (o)verwrite, or (q)uit? ")
        if ans == "o":
            print("Overwriting existing rois file...")
            write_header(rois_file)
        elif ans == "q":
            sys.exit(1)
    else:
        write_header(rois_file)

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
        if not args.no_body:
            # convert to a 2D compressed matrix, because we have a lot of 0's for the
            # background
            compressed = sparse.csr_matrix(np.squeeze(raw_output_up_))
            fname = os.path.splitext(os.path.basename(str(jpg_path)))[0]
            out = os.path.join(args.img_path, args.body_dir, fname)
            sparse.save_npz(out, compressed)

        # == Next, save the ROIs ==
        if not args.no_rois:
            img_id = extract_nums_only(fname)
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
                            (img_id, str(min_x), str(min_y), str(max_x), str(max_y), "\n")
                        )
                    )

        # Save an image of the mask for our own reference every 1000 steps
        if not args.no_vis and step % args.visualize_step == 0:
            preds = np.expand_dims(raw_output_up_, axis=3)
            msk = decode_labels(preds, num_classes=args.num_classes)
            # the mask
            im = Image.fromarray(msk[0])
            # # Save the mask separately
            # jpg_path = str(jpg_path).split('/')[-1].split('.')[0]
            # out = os.path.join(args.vis_dir, jpg_path + '.png')
            # im.save(out)
            # Save the mask with background
            img_orig = Image.open(jpg_path)
            # create the final result using the mask and the original
            img = np.array(im) * 0.9 + np.array(img_orig) * 0.7
            # clip surpassed colors
            img[img > 255] = 255
            img = Image.fromarray(np.uint8(img))
            out = os.path.join(args.img_path, args.vis_dir, fname + ".png")
            img.save(out)
            # # print('Image processed {}.png'.format(jpg_path))
        t.set_description("Finished " + fname)

    total_time = time.time() - start_time
    print(f"The output files have been saved to {args.img_path}/{args.body_dir}")
    print(f"It took {total_time / num_steps} sec on each image.")


if __name__ == "__main__":
    main()
