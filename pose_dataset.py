import numpy as np

import matplotlib
# matplotlib.use('Agg')

from gan.dataset import UGANDataset
import pose_utils
import pose_transform

from skimage.io import imread
import pandas as pd
import os

class PoseHMDataset(UGANDataset):
    def __init__(self, test_phase=False, **kwargs):
        super(PoseHMDataset, self).__init__(kwargs['batch_size'], None)
        self._test_phase = test_phase

        self.use_body_mask = kwargs['use_body_mask']
        self.use_mask = kwargs['use_mask']
        self.num_mask = kwargs['num_mask']
        self._sigma = kwargs['sigma']
        self._num_landmark = kwargs['num_landmarks']
        self._number_of_batches = kwargs['number_of_batches']
        self._batch_size = 1 if self._test_phase else kwargs['batch_size']
        self._image_size = kwargs['image_size']
        self._images_dir_train = kwargs['images_dir_train']
        self._images_dir_test = kwargs['images_dir_test']
        self.fat = kwargs['fat']

        self._bg_images_dir_train = kwargs['bg_images_dir_train']
        self._bg_images_dir_test = kwargs['bg_images_dir_test']

        self._pairs_file_train = pd.read_csv(kwargs['pairs_file_train'])
        self._pairs_file_test = pd.read_csv(kwargs['pairs_file_test'])

        self._annotations_file_test = pd.read_csv(kwargs['annotations_file_train'], sep=':')
        self._annotations_file_train = pd.read_csv(kwargs['annotations_file_test'], sep=':')

        self._annotations_file = pd.concat([self._annotations_file_test, self._annotations_file_train],
                                           axis=0, ignore_index=True)

        self._annotations_file = self._annotations_file.set_index('name')

        self._use_input_pose = kwargs['use_input_pose']
        self._disc_type = kwargs['disc_type']
        self._tmp_pose = kwargs['tmp_pose_dir']
        self._pose_rep_type = kwargs['pose_rep_type']
        self._cache_pose_rep = kwargs['cache_pose_rep']

        self._test_data_index = 0

        if not os.path.exists(self._tmp_pose):
            os.makedirs(self._tmp_pose)

        print ("Number of images: %s" % len(self._annotations_file))
        print ("Number of pairs train: %s" % len(self._pairs_file_train))
        print ("Number of pairs test: %s" % len(self._pairs_file_test))

        self._batches_before_shuffle = int(self._pairs_file_train.shape[0] // self._batch_size)

    def number_of_batches_per_epoch(self):
        return self._number_of_batches

    def number_of_batches_per_validation(self):
        return len(self._pairs_file_test) // self._batch_size

    def compute_pose_map_batch(self, pair_df, direction):
        assert direction in ['to', 'from']
        batch = np.empty([self._batch_size] + list(self._image_size) + [self._num_landmark])
        i = 0
        for _, p in pair_df.iterrows():
            row = self._annotations_file.loc[p[direction]]
            if self._cache_pose_rep:
                file_name = self._tmp_pose + p[direction] + self._pose_rep_type + '.npy'
                if os.path.exists(file_name):
                    pose = np.load(file_name)
                else:
                    kp_array = pose_utils.load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])
                    if self._pose_rep_type == 'hm':
                        pose = np.transpose(pose_utils.cords_to_map(kp_array, self._image_size),[1,2,0])
                    else:
                        pose = pose_transform.make_stickman(kp_array, self._image_size)
                    np.save(file_name, pose)
            else:
                    kp_array = pose_utils.load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])
                    if self._pose_rep_type == 'hm':
                        pose = np.transpose(pose_utils.cords_to_map(kp_array, self._image_size),[1,2,0])
                    else:
                        pose = pose_transform.make_stickman(kp_array, self._image_size)
            batch[i] = pose
            i += 1
        return batch

    def compute_cord_warp_batch(self, pair_df, validation=False):
        batch = [np.empty([self._batch_size] + [self.num_mask, 8]),
                 np.empty([self._batch_size, self.num_mask] + list(self._image_size)),
                 np.empty([self._batch_size] + [self.num_mask, 8]),
                 np.empty([self._batch_size, self.num_mask] + list(self._image_size)),
                 np.empty([self._batch_size] + [self.num_mask, 8]),
                 np.empty([self._batch_size, self.num_mask] + list(self._image_size))]
        i = 0
        for _, p in pair_df.iterrows():
            fr = self._annotations_file.loc[p['from']]
            to = self._annotations_file.loc[p['to']]
            kp_array1 = pose_utils.load_pose_cords_from_strings(fr['keypoints_y'],
                                                                fr['keypoints_x'])
            kp_array2 = pose_utils.load_pose_cords_from_strings(to['keypoints_y'],
                                                                to['keypoints_x'])
            if validation:
                npy_path_from = os.path.join(self._images_dir_test, p['from'])
                npy_path_from = npy_path_from[:-3]+'npy'
                npy_path_to = os.path.join(self._images_dir_test, p['to'])
                npy_path_to = npy_path_to[:-3] + 'npy'
            else:
                npy_path_from = os.path.join(self._images_dir_train, p['from'])
                npy_path_from = npy_path_from[:-3]+'npy'
                npy_path_to = os.path.join(self._images_dir_train, p['to'])
                npy_path_to = npy_path_to[:-3] + 'npy'
            batch[0][i] = pose_transform.affine_transforms(kp_array1, kp_array2, self._image_size, self.use_body_mask)
            batch[1][i] = pose_transform.pose_masks(kp_array2, self._image_size, self.use_body_mask, self.use_mask, npy_path_to, self.fat)
            batch[2][i] = np.c_[np.ones([10,1]),np.zeros([10,3]),np.ones([10,1]),np.zeros([10,3])]
            batch[3][i] = batch[1][i]
            batch[4][i] = pose_transform.affine_transforms(kp_array2, kp_array1, self._image_size, self.use_body_mask)
            batch[5][i] = pose_transform.pose_masks(kp_array1, self._image_size, self.use_body_mask, self.use_mask, npy_path_from, self.fat)

            i += 1
        return batch

    def _preprocess_image(self, image):
        return (image / 255 - 0.5) * 2

    def _deprocess_image(self, image):
        return (255 * (image + 1) / 2).astype('uint8')

    def load_image_batch(self, pair_df, direction='from'):
        assert direction in ['to', 'from']
        batch = np.empty([self._batch_size] + list(self._image_size) + [3])
        i = 0
        for _, p in pair_df.iterrows():
            if os.path.exists(os.path.join(self._images_dir_train, p[direction])):
                batch[i] = imread(os.path.join(self._images_dir_train, p[direction]))
            else:
                batch[i] = imread(os.path.join(self._images_dir_test, p[direction]))
            i += 1
        return self._preprocess_image(batch)

    def load_bg(self, pair_df):
        batch = np.empty([self._batch_size] + list(self._image_size) + [3])
        i = 0
        for _, p in pair_df.iterrows():
            name = p['to'].replace('.jpg', '_BG.jpg') 
            #print os.path.join(self._images_dir_train, name)
            if os.path.exists(os.path.join(self._bg_images_dir_train, name)):
                batch[i] = imread(os.path.join(self._bg_images_dir_train, name))
            else:
                batch[i] = imread(os.path.join(self._bg_images_dir_test, name))
            i += 1
        return self._preprocess_image(batch)

    def load_batch(self, index, for_discriminator, validation=False):
        if validation:
            pair_df = self._pairs_file_test.iloc[index]
        else:
            pair_df = self._pairs_file_train.iloc[index]
        result = [self.load_image_batch(pair_df, 'from')]
        if self._use_input_pose:
            result.append(self.compute_pose_map_batch(pair_df, 'from'))
        result.append(self.load_image_batch(pair_df, 'to'))
        result.append(self.compute_pose_map_batch(pair_df, 'to'))

        result.append(result[-2])
        result += self.compute_cord_warp_batch(pair_df, validation)

        return result

    # # def load_batch(self, index, for_discriminator, validation=False):
    # def load_batch(self, pair_df):
    #     result = [self.load_image_batch(pair_df, 'from')]
    #     if self._use_input_pose:
    #         result.append(self.compute_pose_map_batch(pair_df, 'from'))
    #     result.append(self.load_image_batch(pair_df, 'to'))
    #     result.append(self.compute_pose_map_batch(pair_df, 'to'))
    #
    #     if self._use_bg:
    #         result.append(self.load_image_batch(pair_df, 'to'))
    #         # result.append(self.load_bg(pair_df))
    #
    #     if self._warp_skip != 'none' and (not self._disc_type == 'warp'):
    #         result += self.compute_cord_warp_batch(pair_df)
    #
    #     # if self._use_bg:#####
    #     #     result = result[:-4]#######
    #
    #     return result

    def next_generator_sample(self):
        index = self._next_data_index()

        return self.load_batch(index, False)

    # def next_generator_sample(self):
    #     if self.use_mask==1:
    #         flag = 0
    #         npy_path_from = '.'
    #         npy_path_to = '.'
    #         while (not os.path.exists(npy_path_from)) or (not os.path.exists(npy_path_to)) or flag==0:
    #             flag = 1
    #             index = self._next_data_index()
    #             pair_df = self._pairs_file_train.iloc[index]
    #             npy_path_from = os.path.join(self._images_dir_train, pair_df['from'])
    #             npy_path_from = npy_path_from[:-3] + 'npy'
    #             npy_path_to = os.path.join(self._images_dir_train, pair_df['to'])
    #             npy_path_to = npy_path_to[:-3] + 'npy'
    #     else:
    #         index = self._next_data_index()
    #         pair_df = self._pairs_file_train.iloc[index]
    #     return self.load_batch(pair_df)

    def path_join_batch(self,path, list_path):
        paths = []
        for ids in range(self._batch_size):
            paths.append(path)
        return paths

    def next_generator_sample_test(self, with_names=False):
        index = np.arange(self._test_data_index, self._test_data_index + self._batch_size)
        index = index % self._pairs_file_test.shape[0]
        batch = self.load_batch(index, False, True)
        names = self._pairs_file_test.iloc[index]
        self._test_data_index += self._batch_size
        if with_names:
            return batch, names
        else:
            return batch

    # def next_generator_sample_test(self, with_names=False):
    #     if self.use_mask==1:
    #         flag = 0
    #         npy_path_from = '.'
    #         npy_path_to = '.'
    #         while (not os.path.exists(npy_path_from)) or (not os.path.exists(npy_path_to)) or flag==0:
    #             flag = 1
    #             index = np.arange(self._test_data_index, self._test_data_index + self._batch_size)
    #             index = index % self._pairs_file_test.shape[0]
    #             pair_df = self._pairs_file_test.iloc[index]
    #             npy_path_from = os.path.join(self.tile_list(self._images_dir_train), pair_df['from'])
    #             npy_path_from = npy_path_from[:-3] + 'npy'
    #             npy_path_to = os.path.join(self._images_dir_train, pair_df['to'])
    #             npy_path_to = npy_path_to[:-3] + 'npy'
    #     else:
    #         index = np.arange(self._test_data_index, self._test_data_index + self._batch_size)
    #         index = index % self._pairs_file_test.shape[0]
    #         pair_df = self._pairs_file_test.iloc[index]
    #
    #     batch = self.load_batch(pair_df)
    #     names = pair_df
    #     self._test_data_index += self._batch_size
    #
    #     if with_names:
    #         return batch, names
    #     else:
    #         return batch

    # def next_discriminator_sample(self):
    #     if self.use_mask==1:
    #         flag = 0
    #         npy_path_from = '.'
    #         npy_path_to = '.'
    #         while (not os.path.exists(npy_path_from)) or (not os.path.exists(npy_path_to)) or flag == 0:
    #             flag = 1
    #             index = self._next_data_index()
    #             pair_df = self._pairs_file_train.iloc[index]
    #             npy_path_from = os.path.join(self._images_dir_train, pair_df['from'])
    #             npy_path_from = npy_path_from[:-3] + 'npy'
    #             npy_path_to = os.path.join(self._images_dir_train, pair_df['to'])
    #             npy_path_to = npy_path_to[:-3] + 'npy'
    #     else:
    #         index = self._next_data_index()
    #         pair_df = self._pairs_file_train.iloc[index]
    #     return self.load_batch(pair_df)

    def next_discriminator_sample(self):
        index = self._next_data_index()

        return self.load_batch(index, True)

    def _shuffle_data(self):
        self._pairs_file_train = self._pairs_file_train.sample(frac=1)
        
    def display(self, output_batch, input_batch):
        row = self._batch_size
        col = 1

        tg_app = self._deprocess_image(input_batch[0])
        tg_app = super(PoseHMDataset, self).display(tg_app, None, row=row, col=col)
        src_pose = input_batch[1]
        tg_pose = input_batch[3 if self._use_input_pose else 2]
        tg_img = input_batch[2 if self._use_input_pose else 1]
        tg_img = self._deprocess_image(tg_img)
        tg_img = super(PoseHMDataset, self).display(tg_img, None, row=row, col=col)
        mask_inp = input_batch[-1]
        mask_inp = np.sum(mask_inp, 1)
        mask_inp = np.clip(mask_inp, 0, 1)
        mask_inp = 255 * mask_inp[..., np.newaxis]
        mask_inp = np.tile(mask_inp, (1, 1, 1, 3)).astype(int)
        mask_inp = super(PoseHMDataset, self).display(mask_inp, None, row=row, col=col)
        mask_inp = (mask_inp * tg_app / 255).astype(int)

        mask_out = input_batch[-5]
        mask_out = np.sum(mask_out, 1)
        mask_out = np.clip(mask_out, 0, 1)
        mask_out = 255 * mask_out[..., np.newaxis]
        mask_out = np.tile(mask_out, (1, 1, 1, 3)).astype(int)
        mask_out = super(PoseHMDataset, self).display(mask_out, None, row=row, col=col)
        mask_out = (mask_out * tg_img / 255).astype(int)

        mask_orig = input_batch[-1]
        mask0 = []
        for id in range(10):
            mask = mask_orig[:,id,:,:]
            # if self.use_body_mask:
            #     mask = np.sum(mask, 1)
            #     # mask = np.sum(mask[:,1:,:,:],1)
            # else:
            #     mask = np.sum(mask, 1)
            mask = np.clip(mask,0,1)
            mask = 255*mask[..., np.newaxis]
            mask = np.tile(mask, (1,1,1,3)).astype(int)
            # mask = np.transpose(mask,[0,3,1,2])
            mask = super(PoseHMDataset, self).display(mask, None, row=row, col=col)
            mask0.append((mask*tg_app/255).astype(int))


        res_img = self._deprocess_image(output_batch[2 if self._use_input_pose else 1])


        if self._pose_rep_type == 'hm':
            pose_images = np.array([pose_utils.draw_pose_from_map(pose)[0] for pose in tg_pose])
        else:
            pose_images =  (255 * tg_pose).astype('uint8')
        tg_pose = super(PoseHMDataset, self).display(pose_images, None, row=row, col=col)

        if self._pose_rep_type == 'hm':
            pose_images = np.array([pose_utils.draw_pose_from_map(pose)[0] for pose in src_pose])
        else:
            pose_images =  (255 * src_pose).astype('uint8')
        src_pose = super(PoseHMDataset, self).display(pose_images, None, row=row, col=col)

        res_img = super(PoseHMDataset, self).display(res_img, None, row=row, col=col)

        return np.concatenate(np.array([tg_app, tg_pose, src_pose, mask_inp, mask_out]+ mask0+ [tg_img, res_img]), axis=1)
