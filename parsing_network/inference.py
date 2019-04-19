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

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
NUM_CLASSES = 7
DATA_LIST = './dataset/dance.txt'
SAVE_DIR = './output/'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("img_path", type=str,
                        help="Path to the RGB image file folder.")
    parser.add_argument("model_weights", type=str,
                        help="Path to the file with model weights.")
    parser.add_argument("--data_list", type=str, default=DATA_LIST,
                        help="Path to the image list.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
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
            None, # No defined input size.
            False, # No random scale.
            False, # No random mirror.
            255,
            IMG_MEAN,
            coord)
        image, label = reader.image, reader.label
        title = reader.queue[0]
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.
    
    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    before_argmax = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(before_argmax, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    
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
    # Perform inference.
    t = trange(num_steps, desc="Inference progress", unit="img")
    for step in t:
        # Compressed version
        raw_output_up_, jpg_path = sess.run([raw_output_up, title])
        # convert to a compressed matrix, because we have a lot of 0's for the
        # background
        compressed = sparse.csr_matrix(np.squeeze(raw_output_up_))
        jpg_path = str(jpg_path).split('/')[-1].split('.')[0]
        out = os.path.join(args.save_dir, jpg_path)
        sparse.save_npz(out, compressed)

        # # PNG version
        # preds, jpg_path = sess.run([pred, title])
        # msk = decode_labels(preds, num_classes=args.num_classes)
        # # the mask
        # im = Image.fromarray(msk[0])
        # # save the mask
        # jpg_path = str(jpg_path).split('/')[-1].split('.')[0]
        # out = os.path.join(args.save_dir, jpg_path + '.png')
        # im.save(out)
        t.set_description("Finished " + jpg_path)

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
    print(('The output files have been saved to {}'.format(args.save_dir)))
    print(('It took {} sec on each image.'.format(total_time/num_steps)))
    
if __name__ == '__main__':
    main()
