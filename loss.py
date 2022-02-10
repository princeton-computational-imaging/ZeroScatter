import tensorflow as tf
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk

def discriminator_loss(disc_real_output, disc_generated_output, dloss_weight = 0.5):
    """BinaryCrossentropy loss for discriminator"""
    loss_fun =  tf.keras.losses.BinaryCrossentropy(from_logits = True)
    real_loss = tf.zeros(shape = ())
    for output in disc_real_output:
        real_loss = real_loss + loss_fun(tf.ones_like(output), output)
    generated_loss = tf.zeros(shape = ())
    for output in disc_generated_output:
        generated_loss = generated_loss + loss_fun(tf.zeros_like(output), output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss*dloss_weight

def generator_loss(gen_output,gloss_weight = 0.5):
    """adversarial loss"""
    if gloss_weight > 0:
        loss_fun =  tf.keras.losses.BinaryCrossentropy(from_logits = True)
        gan_loss = tf.zeros(shape = ())
        for output in gen_output:
            gan_loss = gan_loss + loss_fun(tf.ones_like(output), output)
    else:
        gan_loss = 0
    return gan_loss*gloss_weight

def P_loss(G_img, gt_img, vgg_model, loss_weight):
    """perceptual loss"""
    if loss_weight == 0:
        return 0
    else:
        preprocessed_G_img  = tf.keras.applications.vgg19.preprocess_input(G_img*255.0)
        preprocessed_gt_img = tf.keras.applications.vgg19.preprocess_input(gt_img*255.0)

        G_layer_outs = vgg_model(preprocessed_G_img)
        gt_layer_outs = vgg_model(preprocessed_gt_img)

        perc_loss = tf.add_n([tf.reduce_mean(tf.abs(G_layer_out-gt_layer_out)) 
                              for G_layer_out, gt_layer_out in zip(G_layer_outs, gt_layer_outs)])
        return perc_loss* loss_weight

def ent_mask(true_gated, size = 7, thres = 3):
    ent = entropy(tf.squeeze(true_gated[...,0]), disk(size))
    mask = ent > thres
    mask = np.repeat(mask[None,...,None], 3, axis = -1)
    return tf.math.logical_and(mask, true_gated < 0.95)

def dist_match(true_gated, fake_gated, mask):
    masked_true_gated = tf.boolean_mask(true_gated, mask)
    masked_fake_gated = tf.boolean_mask(fake_gated, mask)
    true_gated_mean = tf.reduce_mean(masked_true_gated)
    true_gated_std = tf.math.reduce_std(masked_true_gated)
    fake_gated_std = tf.math.reduce_std(masked_fake_gated)
    fake_gated_mean = tf.reduce_mean(masked_fake_gated)
    fake_gated_adj = tf.clip_by_value(true_gated_mean + (fake_gated - fake_gated_mean)/fake_gated_std * true_gated_std,0,1)
    masked_true_gated = tf.where(mask, true_gated, tf.zeros_like(true_gated))
    masked_fake_gated = tf.where(mask, fake_gated, tf.zeros_like(true_gated))
    masked_fake_gated_adj = tf.where(mask, fake_gated_adj, tf.zeros_like(true_gated))
    return masked_true_gated, masked_fake_gated, tf.reduce_mean(tf.abs(masked_fake_gated - masked_fake_gated_adj))

def gated_supervision_loss(img_left_out, true_gated, rgb2gated, vgg_model, gloss_weight = 1):
    if gloss_weight == 0:
        return 0
    else:
        fake_gated = rgb2gated(img_left_out, training = False)
        mask = ent_mask(true_gated)
        masked_true_gated, masked_fake_gated, score = dist_match(true_gated, fake_gated, mask)
        if score > 0.02:
            loss = 0.001 * P_loss(masked_fake_gated, masked_true_gated, vgg_model, 1) / tf.reduce_mean(tf.cast(mask, tf.float32))
        elif score > 0.01:
            loss = 0.01 * P_loss(masked_fake_gated, masked_true_gated, vgg_model, 1) / tf.reduce_mean(tf.cast(mask, tf.float32))
        else:
            loss = 0.02 * P_loss(masked_fake_gated, masked_true_gated, vgg_model, 1) / tf.reduce_mean(tf.cast(mask, tf.float32))
        return loss * gloss_weight

def spatial_loss(G_img, gt_img, loss_weight):
    if loss_weight == 0:
        return 0
    else:
        def spatial_gradient(x):
            diag_down = x[:, 1:, 1:, :] - x[:, :-1, :-1, :]
            dv = x[:, 1:, :, :] - x[:, :-1, :, :]
            dh = x[:, :, 1:, :] - x[:, :, :-1, :]
            diag_up = x[:, :-1, 1:, :] - x[:, 1:, :-1, :]
            return [dh, dv, diag_down, diag_up]
        gx = spatial_gradient(G_img)
        gy = spatial_gradient(gt_img)
        loss = 0
        for xx, yy in zip(gx, gy):
            loss = loss + tf.reduce_mean(tf.abs(xx - yy))
        return loss * loss_weight

def paired_supervision_loss(real_image, same_image, vgg_model, vgg_weight, spatial_weight, ploss_weight = 5):
    if ploss_weight == 0:
        return 0
    else:
        loss = tf.reduce_mean(tf.abs(real_image - same_image)) + P_loss(same_image, real_image, vgg_model, vgg_weight) + spatial_loss(same_image, real_image, spatial_weight)
        return loss * ploss_weight

def TV_reg_loss(image, loss_weight = 1):
    if loss_weight > 0:
        return 1e-5 * tf.reduce_sum(tf.image.total_variation(image)) * loss_weight
    else:
        return 0

def stereo_loss(out_l, out_r_warped, in_l, loss_weight):
    M = tf.reduce_sum(tf.abs(out_r_warped - in_l)[...,1:], axis = -1, keepdims=True)
    loss = tf.reduce_mean((tf.exp(-10*M)*tf.abs(out_r_warped - out_l)))
    return loss*loss_weight

def Edge_loss(G_img, gt_img, loss_weight):
    if loss_weight == 0:
        return 0
    else:
        def high_pass_avg(image):
            x_var = image[:,:,1:,:] - image[:,:,:-1,:]
            y_var = image[:,1:,:,:] - image[:,:-1,:,:]
            return tf.reduce_mean(tf.abs(x_var)) + tf.reduce_mean(tf.abs(y_var))
        return tf.abs(high_pass_avg(G_img) - high_pass_avg(gt_img)) * loss_weight

def left2right(input_images, x_offset, wrap_mode='border', name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        with tf.name_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
        return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        with tf.name_scope('_interpolate'):

            # handle both texture border types
            _edge_size = 0
            if _wrap_mode == 'border':
                _edge_size = 1
                im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
                x = x + _edge_size
                y = y + _edge_size
            elif _wrap_mode == 'edge':
                _edge_size = 0
            else:
                return None

            x = tf.clip_by_value(x, 0.0,  _width_f - 1 + 2 * _edge_size)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f,  _width_f - 1 + 2 * _edge_size), tf.int32)

            dim2 = (_width + 2 * _edge_size)
            dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
            base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
            base_y0 = base + y0 * dim2
            idx_l = base_y0 + x0
            idx_r = base_y0 + x1

            im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

            pix_l = tf.gather(im_flat, idx_l)
            pix_r = tf.gather(im_flat, idx_r)

            weight_l = tf.expand_dims(x1_f - x, 1)
            weight_r = tf.expand_dims(x - x0_f, 1)

        return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):
        with tf.name_scope('transform'):
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width),
                                       tf.linspace(0.0 , _height_f - 1.0 , _height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])

            x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * _width_f

            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            output = tf.reshape(
                input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
        return output

    with tf.name_scope(name):
        _num_batch    = tf.shape(input_images)[0]
        _height       = tf.shape(input_images)[1]
        _width        = tf.shape(input_images)[2]
        _num_channels = tf.shape(input_images)[3]

        _height_f = tf.cast(_height, tf.float32)
        _width_f  = tf.cast(_width,  tf.float32)

        _wrap_mode = wrap_mode

        output = _transform(input_images, x_offset)
    return output

