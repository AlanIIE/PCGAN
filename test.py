from __future__ import absolute_import, division, print_function
import os

from conditional_gan import make_generator
import cmd
from pose_dataset import PoseHMDataset

from gan.inception_score import get_inception_score
# from gan import fid

from skimage.io import imread, imsave
from skimage.measure import compare_ssim

import numpy as np
import pandas as pd

from tqdm import tqdm
import re
import pose_utils

import numpy as np
import os
import gzip, pickle
import tensorflow as tf
from scipy.misc import imread
from scipy import linalg
import pathlib
import urllib

def l1_score(generated_images, reference_images):
    score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        score = np.abs(2 * (reference_image/255.0 - 0.5) - 2 * (generated_image/255.0 - 0.5)).mean()
        score_list.append(score)
    return np.mean(score_list)


def ssim_score(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        ssim = compare_ssim(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
    return np.mean(ssim_score_list)


def save_images(input_images, pose_inp_array, out_pose, pose_out_array, inp_pose, target_images, generated_images, names, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for images in zip(input_images, pose_inp_array, out_pose, pose_out_array, inp_pose, target_images, generated_images, names):
        # name0 = images[-1]
        # name0 = [name0[1],name0[1]]
        # res_name = str('_'.join(name0)) + '.png'
        # imsave(os.path.join(output_folder, res_name), images[-2])
        res_name = str('_'.join(images[-1])) + '.png'
        imsave(os.path.join(output_folder, res_name), np.concatenate(images[:-1], axis=1))


def create_masked_image(names, images, annotation_file):
    import pose_utils
    masked_images = []
    df = pd.read_csv(annotation_file, sep=':')
    for name, image in zip(names, images):
        to = name[1]
        ano_to = df[df['name'] == to].iloc[0]

        kp_to = pose_utils.load_pose_cords_from_strings(ano_to['keypoints_y'], ano_to['keypoints_x'])

        mask = pose_utils.produce_ma_mask(kp_to, image.shape[:2])
        masked_images.append(image * mask[..., np.newaxis])

    return masked_images


def load_generated_images(images_folder):
    input_images = []
    target_images = []
    generated_images = []
    names = []
    for img_name in os.listdir(images_folder):
        img = imread(os.path.join(images_folder, img_name))
        h = img.shape[1] / 3
        input_images.append(img[:, :h])
        target_images.append(img[:, h:2*h])
        generated_images.append(img[:, 2*h:])

        m = re.match(r'([A-Za-z0-9_]*.jpg)_([A-Za-z0-9_]*.jpg)', img_name)
        fr = m.groups()[0]
        to = m.groups()[1]
        names.append([fr, to])

    return input_images, target_images, generated_images, names


def generate_images(dataset, generator,  use_input_pose):
    input_images = []
    target_images = []
    generated_images = []
    input_pose = []
    out_pose = []
    pose_inp = []
    pose_out = []
    names = []

    def deprocess_image(img):
        return (255 * ((img + 1) / 2.0)).astype(np.uint8)

    for _ in tqdm(range(dataset._pairs_file_test.shape[0])):
        batch, name = dataset.next_generator_sample_test(with_names=True)
        out = generator.predict(batch)
        input_images.append(deprocess_image(batch[0]))
        out_index = 2 if use_input_pose else 1
        target_images.append(deprocess_image(batch[out_index]))
        pose_inp.append(np.array([pose_utils.draw_pose_from_map(pose)[0] for pose in batch[1 if use_input_pose else 0]]))
        pose_out.append(
            np.array([pose_utils.draw_pose_from_map(pose)[0] for pose in batch[3 if use_input_pose else 2]]))
        # pose_inp.append((255 * batch[1 if use_input_pose else 0]).astype('uint8'))
        # pose_out.append((255 * batch[3 if use_input_pose else 2]).astype('uint8'))

        # mask = np.zeros(batch[-1][:,0,:,:].shape)
        # for i in range(10):
        #     mask +=batch[-1][:,i,:,:]*10
        # mask = mask/np.max(mask)
        mask = np.sum(batch[-1], 1)
        mask = mask / np.max(mask)
        # mask = np.clip(np.sum(batch[-1], 1), 0, 1)
        mask = 255 * mask[..., np.newaxis]
        mask = np.tile(mask, (1, 1, 1, 3)).astype(np.uint8)
        input_pose.append(mask)

        mask = np.sum(batch[-5], 1)
        mask = mask / np.max(mask)
        # mask = np.clip(np.sum(batch[-5], 1), 0, 1)
        mask = 255 * mask[..., np.newaxis]
        mask = np.tile(mask, (1, 1, 3)).astype(np.uint8)
        out_pose.append(mask)
        generated_images.append(deprocess_image(out[out_index]))
        names.append([name.iloc[0]['from'], name.iloc[0]['to']])

    input_array = np.concatenate(input_images, axis=0)
    target_array = np.concatenate(target_images, axis=0)
    out_array = np.concatenate(out_pose,axis=0)
    input_pose = np.concatenate(input_pose,axis=0)
    generated_array = np.concatenate(generated_images, axis=0)
    pose_inp_array = np.concatenate(pose_inp, axis=0)
    pose_out_array = np.concatenate(pose_out, axis=0)
    # return input_array*(input_pose/255).astype(np.uint8), pose_inp_array, input_pose, pose_out_array, out_array, (1-out_array/255).astype(np.uint8)*target_array, generated_array, names
    return input_array, pose_inp_array, input_pose, pose_out_array, out_array, target_array, generated_array, names


# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     """Numpy implementation of the Frechet Distance.
#     The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
#     and X_2 ~ N(mu_2, C_2) is
#             d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
#
#     Stable version by Dougal J. Sutherland.
#     Params:
#     -- mu1 : Numpy array containing the activations of the pool_3 layer of the
#              inception net ( like returned by the function 'get_predictions')
#              for generated samples.
#     -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
#                on an representive data set.
#     -- sigma1: The covariance matrix over activations of the pool_3 layer for
#                generated samples.
#     -- sigma2: The covariance matrix over activations of the pool_3 layer,
#                precalcualted on an representive data set.
#     Returns:
#     --   : The Frechet Distance.
#     """
#
#     mu1 = np.atleast_1d(mu1)
#     mu2 = np.atleast_1d(mu2)
#
#     sigma1 = np.atleast_2d(sigma1)
#     sigma2 = np.atleast_2d(sigma2)
#
#     assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
#     assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
#
#     diff = mu1 - mu2
#
#     # product might be almost singular
#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if not np.isfinite(covmean).all():
#         msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
#         # warnings.warn(msg)
#         offset = np.eye(sigma1.shape[0]) * eps
#         covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
#
#     # numerical error might give slight imaginary component
#     if np.iscomplexobj(covmean):
#         if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
#             m = np.max(np.abs(covmean.imag))
#             raise ValueError("Imaginary component {}".format(m))
#         covmean = covmean.real
#
#     tr_covmean = np.trace(covmean)
#
#     return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# -------------------------------------------------------------------------------


def test():
    args = cmd.args()
    if args.load_generated_images:
        print ("Loading images...")
        input_images, target_images, generated_images, names = load_generated_images(args.generated_images_dir)
    else:
        print ("Generate images...")
        from keras import backend as K
        if args.use_dropout_test:
            K.set_learning_phase(1)
        dataset = PoseHMDataset(test_phase=True, **vars(args))
        generator = make_generator(args.image_size, args.use_input_pose, args.warp_agg, args.num_landmarks, args.num_mask)
        assert (args.generator_checkpoint is not None)
        generator.load_weights(args.generator_checkpoint)
        input_images, pose_inp_array, out_pose, pose_out_array, inp_pose, target_images, generated_images, names = generate_images(dataset, generator, args.use_input_pose)
        print ("Save images to %s..." % (args.generated_images_dir, ))
        save_images(input_images, pose_inp_array, out_pose, pose_out_array, inp_pose, target_images, generated_images, names,
                        args.generated_images_dir)

    print ("Compute inception score...")
    inception_score = get_inception_score(generated_images)
    print ("Inception score %s" % inception_score[0])

    # print ("Compute Frechet distance...")
    # fid.create_inception_graph('/tmp/imagenet/classify_image_graph_def.pb')  # load the graph into the current TF graph
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     mu_gen, sigma_gen = fid.calculate_activation_statistics(generated_images, sess, batch_size=100)
    #     mu_real, sigma_real = fid.calculate_activation_statistics(target_images, sess, batch_size=100)
    #
    # fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    # print ("Frechet distance %s" %  fid_value)

    print ("Compute structured similarity score (SSIM)...")
    structured_score = ssim_score(generated_images, target_images)
    print ("SSIM score %s" % structured_score)

    print ("Compute l1 score...")
    norm_score = l1_score(generated_images, target_images)
    print ("L1 score %s" % norm_score)

    print ("Compute masked inception score...")
    generated_images_masked = create_masked_image(names, generated_images, args.annotations_file_test)
    reference_images_masked = create_masked_image(names, target_images, args.annotations_file_test)
    inception_score_masked = get_inception_score(generated_images_masked)
    print ("Inception score masked %s" % inception_score_masked[0])



    # print ("Compute masked Frechet distance...")
    # fid.create_inception_graph('/tmp/imagenet/classify_image_graph_def.pb')  # load the graph into the current TF graph
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     mu_gen, sigma_gen = fid.calculate_activation_statistics(generated_images_masked, sess, batch_size=100)
    #     mu_real, sigma_real = fid.calculate_activation_statistics(reference_images_masked, sess, batch_size=100)
    #
    # fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    # print("Frechet distance masked %s" % fid_value)

    print ("Compute masked SSIM...")
    structured_score_masked = ssim_score(generated_images_masked, reference_images_masked)
    print ("SSIM score masked %s" % structured_score_masked)

    print ("Inception score = %s, masked = %s; SSIM score = %s, masked = %s; l1 score = %s" %
           (inception_score, inception_score_masked, structured_score, structured_score_masked, norm_score))



if __name__ == "__main__":
    test()


