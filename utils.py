import tensorflow as tf
from loss import *

def load_warped_tfrecord(data_path, load_h, load_w, in_h, in_w, training=True):
    """
    Dataloader for adverse weather training data
    img_l: left stereo capture
    img_r: right stereo capture
    img_warped: temporal wraped capture
    img_ref: processed left stereo capture
    disp: per-pixel disparity
    depth: per-pixel depth
    gated: gated imaging capture
    """
    IMAGE_FEATURE_MAP = {
        'img_l': tf.io.FixedLenFeature([], tf.string),
        'img_r': tf.io.FixedLenFeature([], tf.string),
        'img_warped': tf.io.FixedLenFeature([], tf.string),
        'img_ref': tf.io.FixedLenFeature([], tf.string),
        'disp': tf.io.FixedLenFeature([], tf.string),
        'depth': tf.io.FixedLenFeature([], tf.string),
        'gated': tf.io.FixedLenFeature([], tf.string)
    }
    def parse_tfrecord(tfrecord, data_augmentation = True):
        x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
        
        img_l = tf.io.decode_png(x['img_l'], channels=3, dtype=tf.dtypes.uint16)/(2**12-1)
        img_r = tf.io.decode_png(x['img_r'], channels=3, dtype=tf.dtypes.uint16)/(2**12-1)
        img_warped = tf.io.decode_png(x['img_warped'], channels=3, dtype=tf.dtypes.uint16)/(2**12-1)
        img_ref = tf.io.decode_png(x['img_ref'], channels=3, dtype=tf.dtypes.uint16)/(2**12-1)
        gated = tf.io.decode_png(x['gated'], channels=3, dtype=tf.dtypes.uint16)/(2**10-1)
        disp = tf.io.decode_png(x['disp'], channels=3, dtype=tf.dtypes.uint16)/(2**16-1)
        depth = tf.io.decode_png(x['depth'], channels=3, dtype=tf.dtypes.uint16)/(2**16-1)*250

        grouped = tf.concat([img_l, img_r, img_warped, gated, disp, img_ref, depth], axis = -1)
        grouped = tf.image.resize_with_crop_or_pad(grouped, load_h, load_w)
        if in_h*in_w < load_h*load_w:
            grouped = tf.image.random_crop(grouped, size=[in_h, in_w, grouped.shape[-1]])

        img_l = grouped[:,:,0:3]
        img_r = grouped[:,:,3:6]
        img_warped = grouped[:,:,6:9]
        gated = grouped[:,:,9:12]
        disp = grouped[:,:,12]
        img_ref = grouped[:,:,15:18]
        depth = grouped[:,:,18:]
        
        return img_l, img_r, img_warped, img_ref, gated, tf.squeeze(disp), depth

    if training:
        files = tf.data.Dataset.list_files(data_path+'*.tfrecords', shuffle=True)
        dataset = files.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(lambda x: parse_tfrecord(x, True),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(10).repeat().batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = iter(dataset)
    else:
        files = tf.data.Dataset.list_files(data_path+'*.tfrecords', shuffle=False)
        dataset = files.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(lambda x: parse_tfrecord(x, False),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = iter(dataset)       
    return dataset

def load_eval_tfrecord(data_path, load_h, load_w, in_h, in_w, training=False):
    """
    Dataloader for evaluation data (without ground truth)
    img_l: left stereo capture
    img_ref: processed left stereo capture (for comparison purpose)
    """
    IMAGE_FEATURE_MAP = {
        'img_l': tf.io.FixedLenFeature([], tf.string),
        'img_ref': tf.io.FixedLenFeature([], tf.string)
    }
    def parse_tfrecord(tfrecord, data_augmentation = True):
        x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
        
        img_l = tf.io.decode_png(x['img_l'], channels=3, dtype=tf.dtypes.uint16)/(2**12-1)
        img_ref = tf.io.decode_png(x['img_ref'], channels=3, dtype=tf.dtypes.uint16)/(2**12-1)
        
        return img_l, img_ref

    if training:
        files = tf.data.Dataset.list_files(data_path+'*.tfrecords', shuffle=True)
        dataset = files.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(lambda x: parse_tfrecord(x, True),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(10).repeat().batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = iter(dataset)
    else:
        files = tf.data.Dataset.list_files(data_path+'*.tfrecords', shuffle=False)
        dataset = files.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(lambda x: parse_tfrecord(x, False),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = iter(dataset)       
    return dataset

def load_clear_tfrecord(data_path, load_h, load_w, in_h, in_w, training=True):
    """
    Dataloader for clear weather training data, requires
    img_l: left stereo capture
    img_r: right stereo capture
    img_l_foggy0.xxx: synthetic foggy capture with different levels of fog intensity
    """
    IMAGE_FEATURE_MAP = {
        'img_l': tf.io.FixedLenFeature([], tf.string),
        'ref_l': tf.io.FixedLenFeature([], tf.string),
        'img_l_foggy0.00500': tf.io.FixedLenFeature([], tf.string),
        'img_l_foggy0.01000': tf.io.FixedLenFeature([], tf.string),
        'img_l_foggy0.02000': tf.io.FixedLenFeature([], tf.string),
        'img_l_foggy0.04000': tf.io.FixedLenFeature([], tf.string),
        'img_l_foggy0.06000': tf.io.FixedLenFeature([], tf.string),
        'img_l_foggy0.08000': tf.io.FixedLenFeature([], tf.string),
        'img_l_foggy0.10000': tf.io.FixedLenFeature([], tf.string),
    }
    def parse_tfrecord(tfrecord, data_augmentation = True):
        x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
        
        img = tf.io.decode_png(x['img_l'], channels=3, dtype=tf.dtypes.uint16)/(2**12-1)

        ref = tf.gather(tf.io.decode_png(x['ref_l'], channels=3, dtype=tf.dtypes.uint16)/(2**12-1), [2,1,0], axis = -1)

        fog = []
        for b in ['0.00500','0.01000','0.02000','0.04000','0.06000','0.08000','0.10000']:
            fog.append(tf.io.decode_png(x['img_l_foggy'+b], channels=3, dtype=tf.dtypes.uint16)/(2**12-1))

        grouped = tf.concat([img, ref]+ fog, axis = -1)
        grouped = tf.image.resize_with_crop_or_pad(grouped, load_h, load_w)
        if in_h*in_w < load_h*load_w:
            grouped = tf.image.random_crop(grouped, size=[in_h, in_w, grouped.shape[-1]])

        if data_augmentation:
            grouped = tf.image.random_flip_left_right(grouped)
        img = grouped[:,:,0:3]
        ref = grouped[:,:,3:6]
        fog = grouped[:,:,6:]
        return img, ref, fog

    if training:
        files = tf.data.Dataset.list_files(data_path+'*.tfrecords', shuffle=True)
        dataset = files.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(lambda x: parse_tfrecord(x, True),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(10).repeat().batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = iter(dataset)
    else:
        files = tf.data.Dataset.list_files(data_path+'*.tfrecords', shuffle=False)
        dataset = files.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(lambda x: parse_tfrecord(x, False),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = iter(dataset)
    return dataset

def load_fogchamber_tfrecord(data_path, load_h, load_w, in_h, in_w, training=True):
    """
    Dataloader for fog chamber data, requires
    target: ground truth
    fog: foggy capture
    ref: processed foggy capture
    """

    IMAGE_FEATURE_MAP = {
        'target': tf.io.FixedLenFeature([], tf.string),
        'fog': tf.io.FixedLenFeature([], tf.string),
        'ref': tf.io.FixedLenFeature([], tf.string)
    }
    def parse_tfrecord(tfrecord, data_augmentation = True):
        x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
        
        target = tf.io.decode_png(x['target'], channels=3, dtype=tf.dtypes.uint16)/(2**12-1)
        ref = tf.io.decode_png(x['ref'], channels=3, dtype=tf.dtypes.uint16)/(2**12-1)
        fog = tf.io.decode_png(x['fog'], channels=3, dtype=tf.dtypes.uint16)/(2**12-1)

        grouped = tf.concat([target, fog, ref], axis = -1)
        grouped = tf.image.resize_with_crop_or_pad(grouped, load_h, load_w)
        if in_h*in_w < load_h*load_w:
            grouped = tf.image.random_crop(grouped, size=[in_h, in_w, grouped.shape[-1]])

        if data_augmentation:
            grouped = tf.image.random_flip_left_right(grouped)

        target = grouped[:,:,0:3]
        fog = grouped[:,:,3:6]
        fog = grouped[:,:,6:9]
        
        return target, fog, ref

    if training:
        files = tf.data.Dataset.list_files(data_path+'*.tfrecords', shuffle=True)
        dataset = files.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(lambda x: parse_tfrecord(x, True),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(10).repeat().batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = iter(dataset)
    else:
        files = tf.data.Dataset.list_files(data_path+'*.tfrecords', shuffle=False)
        dataset = files.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(lambda x: parse_tfrecord(x, False),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = iter(dataset)       
    return dataset

def load_sequence_tfrecord(data_path, load_h, load_w, in_h, in_w, training=False):
    """
    Dataloader for sequence captured (video) data, requires
    img_l: left stereo capture
    """    
    IMAGE_FEATURE_MAP = {
        'img_l': tf.io.FixedLenFeature([], tf.string),
    }
    def parse_tfrecord(tfrecord):
        x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
        
        img_l = tf.io.decode_png(x['img_l'], channels=3, dtype=tf.dtypes.uint16)/(2**12-1)

        img_l = tf.image.resize_with_crop_or_pad(img_l, load_h, load_w)
        if in_h*in_w < load_h*load_w:
            img_l = tf.image.random_crop(img_l, size=[in_h, in_w, img_l.shape[-1]])

        return img_l

    if training:
        files = tf.data.Dataset.list_files(data_path+'*.tfrecords', shuffle=False)
        dataset = files.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(lambda x: parse_tfrecord(x),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat().batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = iter(dataset)
    else:
        files = tf.data.Dataset.list_files(data_path+'*.tfrecords', shuffle=False)
        dataset = files.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(lambda x: parse_tfrecord(x),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = iter(dataset)       
    return dataset

def load_rgb2gated_tfrecord(data_path, load_h, load_w, in_h, in_w, training=True):
    """
    Dataloader for RGB2Gated training data
    ref_l: processed left stereo capture
    gated: gated imaging capture
    """
    IMAGE_FEATURE_MAP = {
        'ref_l': tf.io.FixedLenFeature([], tf.string),
        'gated': tf.io.FixedLenFeature([], tf.string)
    }
    def parse_tfrecord(tfrecord):
        x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)

        ref_l = tf.gather(tf.io.decode_png(x['ref_l'], channels=3, dtype=tf.dtypes.uint16)/(2**12-1), [2,1,0], axis = -1)
        
        gated = tf.io.decode_png(x['gated'], channels=3, dtype=tf.dtypes.uint16)/(2**10-1)

        grouped = tf.concat([ref_l, gated], axis = -1)
        grouped = tf.image.resize_with_crop_or_pad(grouped, load_h, load_w)
        if in_h*in_w < load_h*load_w:
            grouped = tf.image.random_crop(grouped, size=[in_h, in_w, grouped.shape[-1]])

        ref_l = grouped[:,:,0:3]
        gated = grouped[:,:,3:]
        return ref_l, gated

    if training:
        files = tf.data.Dataset.list_files(data_path+'*.tfrecords', shuffle=True)
        dataset = files.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(lambda x: parse_tfrecord(x),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(10).repeat().batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = iter(dataset)
    else:
        files = tf.data.Dataset.list_files(data_path+'*.tfrecords', shuffle=False)
        dataset = files.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(lambda x: parse_tfrecord(x),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = iter(dataset)
    return dataset

def build_model(network, img_h, img_w):
    if network == 'translation':
        from networks.generator import Translation
        model = Translation([img_h, img_w, 3])
    elif network == 'discriminator':
        from networks.discriminator import Discriminator
        model = Discriminator([img_h, img_w, 3])
    elif network == 'consistency':
        from networks.generator import Consistency
        model = Consistency([img_h, img_w, 3])
    elif network == 'rgb2gated':
        from networks.rgb2gated import RGB2Gated
        model = RGB2Gated([img_h, img_w, 3])
    elif network == 'rgb2gated_disc':
        from networks.rgb2gated import Disc
        model = Disc([img_h, img_w, 3])
    else:  
        model = tf.keras.models.load_model(network)
    return model