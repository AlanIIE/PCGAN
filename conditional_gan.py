from keras.models import Model, Input, Sequential
from keras.layers import Flatten, Concatenate, Activation, Dropout, Dense, Reshape, Multiply, Add, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D, Cropping2D
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
from keras.backend import tf as ktf

from gan.gan import GAN
from gan.layer_utils import content_features_model

from keras.optimizers import Adam
from pose_transform import AffineMaskLayer, SizeTransformLayer, MaskLayer


def block(out, nkernels, down=True, bn=True, dropout=False, leaky=True):
    if leaky:
        out = LeakyReLU(0.2)(out)
    else:
        out = Activation('relu')(out)
    if down:
        out = ZeroPadding2D((1, 1))(out)
        out = Conv2D(nkernels, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(out)
    else:
        out = Conv2DTranspose(nkernels, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(out)
        out = Cropping2D((1, 1))(out)
    if bn:
        out = InstanceNormalization()(out)
    if dropout:
        out = Dropout(0.5)(out)
    return out


def encoder(inps, nfilters=(64, 128, 256, 512, 512, 512), flag=0):
    layers = []
    if len(inps) != 1:
        if flag == 0:
            out = Concatenate(axis=-1)(inps)
        else:
            out = Concatenate(axis=-1, name='bg_concate')(inps)
            tmp = Concatenate(axis=-1)(inps)
    else:
        out = inps[0]
    for i, nf in enumerate(nfilters):
        if i == 0:
            if flag == 0:
                out = Conv2D(nf, kernel_size=(3, 3), padding='same')(out)
            else:
                out = Conv2D(nf, kernel_size=(3, 3), padding='same', name='bg_conv2d_7')(out)
                tmp = Conv2D(nf, kernel_size=(3, 3), padding='same')(out)
        elif i == len(nfilters) - 1:
            out = block(out, nf, bn=False)
        else:
            out = block(out, nf)
        layers.append(out)
    return layers


def decoder(skips, nfilters=(512, 512, 512, 256, 128, 3)):
    out = None
    for i, (skip, nf) in enumerate(zip(skips, nfilters)):
        if 0 < i < 3:
            out = Concatenate(axis=-1)([out, skip])
            out = block(out, nf, down=False, leaky=False, dropout=True)
        elif i == 0:
            out = block(skip, nf, down=False, leaky=False, dropout=True)
        elif i == len(nfilters) - 1:
            out = Concatenate(axis=-1)([out, skip])
            out = Activation('relu')(out)
            out = Conv2D(nf, kernel_size=(3, 3), use_bias=True, padding='same')(out)
        else:
            out = Concatenate(axis=-1)([out, skip])
            out = block(out, nf, down=False, leaky=False)
    out = Activation('tanh')(out)
    return out

def generate_pose(image_size, pose_rep_type, num_landmarks):
    inps = Input(list(image_size)+[3], name='inps')

    cf_model = content_features_model(inps.get_shape().as_list(), 'block3_conv1')
    for layer in cf_model.layers:
        layer.trainable = False

    reference = cf_model(inps)

    feature = Dense(64, input_shape=(reference.get_shape().as_list()[-1],))(reference)

    weight = Flatten(name='flatten')(reference)
    weight = Dense(128)(weight)
    weight = LeakyReLU(0.2)(weight)
    weight = Dense(64)(weight)
    weight = Reshape((1,1,64), input_shape=(64,))(weight)

    out = Multiply()([feature, weight])

    out = SizeTransformLayer(tuple(inps.get_shape().as_list()[:-1]+[64]))(out)
    out = Dense(num_landmarks,input_shape=(64,))(out)

    return Model(inputs=[inps], outputs=[out])

def concatenate_skips(skips_app, skips_pose, warp, image_size, warp_agg, num_mask):
    skips = []

    for i, (sk_app, sk_pose) in enumerate(zip(skips_app, skips_pose)):
        if i < 4:
            out = AffineMaskLayer(num_mask, warp_agg, image_size)([sk_app] + warp)
            out = Concatenate(axis=-1)([out, sk_pose])
        else:
            out = Concatenate(axis=-1)([sk_app, sk_pose])
        skips.append(out)
    return skips

def make_generator(image_size, use_input_pose, warp_agg, num_landmarks, num_mask):

    input_img = Input(list(image_size) + [3], name='input_img')
    output_pose = Input(list(image_size) + [num_landmarks], name='output_pose')
    output_img = Input(list(image_size) + [3], name='output_img')
    nfilters_decoder = (512, 512, 512, 256, 128, 3) if max(image_size) == 128 else (512, 512, 512, 512, 256, 128, 3)
    nfilters_encoder = (64, 128, 256, 512, 512, 512) if max(image_size) == 128 else (64, 128, 256, 512, 512, 512, 512)


    warp_fg = [Input((num_mask, 8), name='warp_fg1'), Input((num_mask, image_size[0], image_size[1]), name='warp_fg2')]
    warp_idt = [Input((num_mask, 8), name='warp_idt1'), Input((num_mask, image_size[0], image_size[1]), name='warp_idt2')]
    warp_bg = [Input((num_mask, 8), name='warp_bg1'), Input((num_mask, image_size[0], image_size[1]), name='warp_bg2')]


    if use_input_pose:
        input_pose = [Input(list(image_size) + [num_landmarks], name='input_pose')]
    else:
        input_pose = []

    bg_img = Input(list(image_size) + [3], name='bg_img')
    bg = [MaskLayer(num_mask, image_size)([bg_img] + warp_fg)]


    enc_app_layers = encoder([input_img] + input_pose, nfilters_encoder)
    flag = 1
    enc_tg_layers = encoder([output_pose] + bg, nfilters_encoder, flag)
    enc_layers = concatenate_skips(enc_app_layers, enc_tg_layers, [warp_fg[0],warp_bg[1]], image_size, warp_agg, num_mask)


    out = decoder(enc_layers[::-1], nfilters_decoder)

    return Model(inputs=[input_img] + input_pose + [output_img, output_pose] + [bg_img] + warp_fg + warp_idt + warp_bg,
                 outputs=[input_img] + input_pose + [out, output_pose] + [bg_img] + warp_fg + warp_idt + warp_bg)


def make_discriminator(image_size, use_input_pose, num_landmarks, num_mask):
    input_img = Input(list(image_size) + [3])
    output_pose = Input(list(image_size) + [num_landmarks])
    input_pose = Input(list(image_size) + [num_landmarks])
    output_img = Input(list(image_size) + [3])
    bg_img = Input(list(image_size) + [3])

    warp = [Input((num_mask, 8)), Input((num_mask, image_size[0], image_size[1]))]
    warp_no1 = [Input((num_mask, 8)), Input((num_mask, image_size[0], image_size[1]))]
    warp_no2 = [Input((num_mask, 8)), Input((num_mask, image_size[0], image_size[1]))]


    if use_input_pose:
        input_pose = [input_pose]
    else:
        input_pose = []

    bg_img0 = [MaskLayer(num_mask, image_size)([bg_img] + warp)]

    out1 = Concatenate(axis=-1)([input_img] + input_pose + [output_img, output_pose] + bg_img0)
    out1 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(out1)
    out1 = block(out1, 128)
    out1 = block(out1, 256)
    out1 = block(out1, 512)
    out1 = block(out1, 1, bn=False)
    out1 = Activation('sigmoid')(out1)
    # out1 = Flatten()(out1)

    out2 = Concatenate(axis=-1)([output_img, output_pose])
    out2 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(out2)
    out2 = block(out2, 128)
    out2 = block(out2, 256)
    out2 = block(out2, 512)
    out2 = block(out2, 1, bn=False)
    out2 = Activation('sigmoid')(out2)
    # out2 = Flatten()(out2)

    out = Add()([out1, out2])
    out = Lambda(lambda x: x / 2)(out)
    # out = Activation('sigmoid')(out)
    out = Flatten()(out)

    # out = Dense(1, kernel_initializer='ones', bias_initializer='zeros', trainable=False)(out)
    # out = out2

    return Model(inputs=[input_img] + input_pose + [output_img, output_pose] + [bg_img] + warp + warp_no1 + warp_no2,
                 outputs=[out])

def total_variation_loss(x, image_size):
    img_nrows, img_ncols = image_size
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def nn_loss(reference, target, neighborhood_size=(3, 3)):
    v_pad = int(neighborhood_size[0] / 2)
    h_pad = int(neighborhood_size[1] / 2)
    val_pad = ktf.pad(reference, [[0, 0], [v_pad, v_pad], [h_pad, h_pad], [0, 0]],
                      mode='CONSTANT', constant_values=-10000)
    reference_tensors = []
    for i_begin in range(0, neighborhood_size[0]):
        i_end = i_begin - neighborhood_size[0] + 1
        i_end = None if i_end == 0 else i_end
        for j_begin in range(0, neighborhood_size[1]):
            j_end = j_begin - neighborhood_size[0] + 1
            j_end = None if j_end == 0 else j_end
            sub_tensor = val_pad[:, i_begin:i_end, j_begin:j_end, :]
            reference_tensors.append(ktf.expand_dims(sub_tensor, -1))
    reference = ktf.concat(reference_tensors, axis=-1)
    target = ktf.expand_dims(target, axis=-1)

    abs = ktf.abs(reference - target)
    norms = ktf.reduce_sum(abs, reduction_indices=[-2])
    loss = ktf.reduce_min(norms, reduction_indices=[-1])

    return loss


class CGAN(GAN):
    def __init__(self, generator, discriminator, l1_penalty_weight, gan_penalty_weight,
                 use_input_pose, image_size, num_landmarks, **kwargs):
        super(CGAN, self).__init__(generator, discriminator, generator_optimizer=Adam(2e-4, 0.5, 0.999),
                                   discriminator_optimizer=Adam(2e-4, 0.5, 0.999), **kwargs)
        generator.summary()
        self._l1_penalty_weight = l1_penalty_weight
        self.generator_metric_names = ['gan_loss','l1_loss']
        self._use_input_pose = use_input_pose
        self._image_size = image_size
        self._num_landmarks = num_landmarks
        self._gan_penalty_weight = gan_penalty_weight


    def _compile_generator_loss(self):
        image_index = 2 if self._use_input_pose else 1

        reference = self._generator_input[image_index]
        target = self._discriminator_fake_input[image_index]
        l1_loss = self._l1_penalty_weight * K.mean(K.abs(reference - target))

        def l1_loss_fn(y_true, y_pred):
            return l1_loss

        def gan_loss_fn(y_true, y_pred):
            loss = super(CGAN, self)._compile_generator_loss()[0](y_true, y_pred)
            return K.constant(0) if self._gan_penalty_weight == 0 else self._gan_penalty_weight * loss

        def generator_loss(y_true, y_pred):
            return gan_loss_fn(y_true, y_pred) + l1_loss_fn(y_true, y_pred)

        return generator_loss, [gan_loss_fn, l1_loss_fn]