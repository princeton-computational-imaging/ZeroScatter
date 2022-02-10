import os
import argparse
import json
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(2)
import random
from matplotlib import pyplot
from utils import *
from loss import *
from networks.foggify import Foggify

def save_performance(G_T, G_C, clear_val, fog_val, snow_val, step, args):

    num_test = len(clear_val) + len(fog_val) + len(snow_val)

    if args.train_consistency:
        pyplot.figure(figsize=(40,30))
        for i in range(len(clear_val)):
            clear_real_now, clear_ref_now, adv_syn_now = clear_val[i]
            beta_idx = 3
            adv_syn_now = adv_syn_now[:,:,:,beta_idx*3:(beta_idx + 1)*3]

            clear_fake_now = G_T(adv_syn_now, training = False)
            clear_fake_now_enhanced = G_C(clear_fake_now, training = False)
            pyplot.subplot(num_test, 4, 4*i + 1)
            pyplot.axis('off')
            if i == 0:
                pyplot.title('Adverse Weather', fontdict={'fontsize':20})
            pyplot.imshow((tf.squeeze(adv_syn_now).numpy()))        
            pyplot.subplot(num_test, 4, 4*i + 2)
            pyplot.axis('off')
            if i == 0:
                pyplot.title('Translation', fontdict={'fontsize':20})
            pyplot.imshow((tf.squeeze(clear_fake_now).numpy())) 
            pyplot.subplot(num_test, 4, 4*i + 3)
            pyplot.axis('off')
            if i == 0:
                pyplot.title('ZeroScatter', fontdict={'fontsize':20})
            pyplot.imshow((tf.squeeze(clear_fake_now_enhanced).numpy()))  
            pyplot.subplot(num_test, 4, 4*i + 4)
            pyplot.axis('off')
            if i == 0:
                pyplot.title('Clear Ref', fontdict={'fontsize':20})
            pyplot.imshow((tf.squeeze(clear_ref_now).numpy()))  
  
        for idx in range(len(fog_val)):
            adv_real_l, *_  = fog_val[idx]

            clear_fake_now = G_T(adv_real_l, training = False)
            clear_fake_now_enhanced = G_C(clear_fake_now, training = False)
            i = idx + len(clear_val) 
            pyplot.subplot(num_test, 4, 4*i+1)
            pyplot.axis('off')
            pyplot.imshow((tf.squeeze(adv_real_l).numpy()))        
            pyplot.subplot(num_test, 4, 4*i+2)
            pyplot.axis('off')
            pyplot.imshow((tf.squeeze(clear_fake_now).numpy()))  
            pyplot.subplot(num_test, 4, 4*i+3)
            pyplot.axis('off')
            pyplot.imshow((tf.squeeze(clear_fake_now_enhanced).numpy()))  

        for idx in range(len(snow_val)):
            adv_real_now= snow_val[idx]

            clear_fake_now = G_T(adv_real_now, training = False)
            clear_fake_now_enhanced = G_C(clear_fake_now, training = False)
            i = idx + len(clear_val) + len(fog_val) 

            pyplot.subplot(num_test, 4, 4*i+1)
            pyplot.axis('off')
            pyplot.imshow((tf.squeeze(adv_real_now).numpy()))        
            pyplot.subplot(num_test, 4, 4*i+2)
            pyplot.axis('off')
            pyplot.imshow((tf.squeeze(clear_fake_now).numpy()))  
            pyplot.subplot(num_test, 4, 4*i+3)
            pyplot.axis('off')
            pyplot.imshow((tf.squeeze(clear_fake_now_enhanced).numpy()))  

    else:
        pyplot.figure(figsize=(30,30))
        for i in range(len(clear_val)):
            clear_real_now, clear_ref_now, adv_syn_now = clear_val[i]
            beta_idx = 3
            adv_syn_now = adv_syn_now[:,:,:,beta_idx*3:(beta_idx + 1)*3]

            clear_fake_now = G_T(adv_syn_now, training = False)
            pyplot.subplot(num_test, 3, 3*i + 1)
            pyplot.axis('off')
            if i == 0:
                pyplot.title('Adverse Weather', fontdict={'fontsize':20})
            pyplot.imshow((tf.squeeze(adv_syn_now).numpy()))        
            pyplot.subplot(num_test, 3, 3*i + 2)
            pyplot.axis('off')
            if i == 0:
                pyplot.title('Translation', fontdict={'fontsize':20})
            pyplot.imshow((tf.squeeze(clear_fake_now).numpy())) 
            pyplot.subplot(num_test, 3, 3*i + 3)
            pyplot.axis('off')
            if i == 0:
                pyplot.title('Clear Ref', fontdict={'fontsize':20})
            pyplot.imshow((tf.squeeze(clear_ref_now).numpy()))  


        for idx in range(len(fog_val)):
            adv_real_l, *_  =fog_val[idx]

            clear_fake_now = G_T(adv_real_l, training = False)
            i = idx + len(clear_val) 
            pyplot.subplot(num_test, 3, 3*i+1)
            pyplot.axis('off')
            pyplot.imshow((tf.squeeze(adv_real_l).numpy()))        
            pyplot.subplot(num_test, 3, 3*i+2)
            pyplot.axis('off')
            pyplot.imshow((tf.squeeze(clear_fake_now).numpy()))  


        for idx in range(len(snow_val)):
            adv_real_now= snow_val[idx]

            clear_fake_now = G_T(adv_real_now, training = False)
            i = idx + len(clear_val) + len(fog_val) 
            pyplot.subplot(num_test, 3, 3*i+1)
            pyplot.axis('off')
            pyplot.imshow((tf.squeeze(adv_real_now).numpy()))        
            pyplot.subplot(num_test, 3, 3*i+2)
            pyplot.axis('off')
            pyplot.imshow((tf.squeeze(clear_fake_now).numpy()))   

    # save plot to file
    filename1 = os.path.join(args.result_path,'plot_%02d.png' % (step/args.save_freq))
    pyplot.savefig(filename1)
    pyplot.close()

def log(G_T, D_clear, D_adv, G_C, RGB2Gated, vgg_misaligned_model, vgg_perpixel_model, clear_val, fog_val, summary_writer, step, args):
    clear_real_now, clear_ref_now, adv_syn_now = clear_val

    beta_idx = 3
    adv_syn_now = adv_syn_now[:,:,:,beta_idx*3:(beta_idx + 1)*3]

    adv_real_l, adv_real_r, adv_warped_l, adv_ref, adv_gated, adv_disp, adv_depth = fog_val

    clear_fake_syn_now = G_T(adv_syn_now, training = False)
    clear_fake_real_now = G_T(adv_real_l, training = False)
    clear_same_now = G_T(clear_real_now, training = False)

    disc_clear_real = D_clear(clear_ref_now, training = False)
    disc_clear_fake = D_clear(tf.concat([clear_fake_real_now, clear_fake_syn_now],axis = 0), training = False)

    disc_adv_real = D_adv(adv_ref, training = False)
    adv_fake = Foggify(clear_fake_real_now, adv_ref, adv_depth)
    disc_adv_fake = D_adv(adv_fake, training = False)

    gen_c_loss = generator_loss(disc_clear_fake, args.adv_loss_weight) + generator_loss(disc_adv_fake, args.adv_loss_weight)
    paired_loss_b2c = paired_supervision_loss(clear_fake_syn_now, clear_ref_now, vgg_perpixel_model, args.perc_loss_weight, args.spatial_loss_weight, args.paired_loss_weight) 
    identity_loss_c = paired_supervision_loss(clear_same_now, clear_ref_now, vgg_perpixel_model, args.perc_loss_weight, args.spatial_loss_weight, args.identity_loss_weight) 
    tv_loss = TV_reg_loss(tf.concat([clear_fake_real_now, clear_fake_syn_now],axis = 0), args.tv_loss_weight)
    disc_c_loss = discriminator_loss(disc_clear_real, disc_clear_fake)
    disc_b_loss = discriminator_loss(disc_adv_real, disc_adv_fake)

    if args.train_consistency:
        clear_fake_real_now_enhanced = G_C(clear_fake_real_now, training = False)
    else:
        clear_fake_real_now_enhanced = clear_fake_real_now

    gated_loss = gated_supervision_loss(clear_fake_real_now_enhanced, adv_gated, RGB2Gated, vgg_misaligned_model, gloss_weight = args.gated_supervision_weight)


    if args.temporal_loss_weight > 0:
        clear_fake_warped_now = G_T(adv_warped_l, training = False)
        temporal_loss = paired_supervision_loss(clear_fake_real_now_enhanced, clear_fake_warped_now, vgg_perpixel_model, args.perc_loss_weight, args.spatial_loss_weight, args.temporal_loss_weight) 
    else:
        temporal_loss = 0

    if args.stereo_loss_weight > 0:
        in_r_warped = left2right(adv_real_r, -1 * adv_disp)
        out_r_warped = G_T(in_r_warped, training = False)
        stereo_consistency_loss = stereo_loss(clear_fake_real_now_enhanced, out_r_warped, adv_real_l, args.stereo_loss_weight)
    else:
        stereo_consistency_loss = 0

    with summary_writer.as_default():
        tf.summary.scalar(name = 'loss/gen_loss', data = gen_c_loss, step=step)
        tf.summary.scalar(name = 'loss/identity_loss', data = identity_loss_c, step=step)
        tf.summary.scalar(name = 'loss/gated_loss', data = gated_loss, step=step)
        tf.summary.scalar(name = 'loss/paired_loss', data = paired_loss_b2c, step=step)
        tf.summary.scalar(name = 'loss/temporal_loss', data = temporal_loss, step=step)
        tf.summary.scalar(name = 'loss/tv_loss', data = tv_loss, step=step)
        tf.summary.scalar(name = 'loss/stereo_loss', data = stereo_consistency_loss, step=step)
        tf.summary.scalar(name = 'loss/Dc_loss'    , data = disc_c_loss   , step=step)
        tf.summary.scalar(name = 'loss/Db_loss'    , data = disc_b_loss   , step=step)

def train_translation_step(G_T, D_clear, D_adv, RGB2Gated, vgg_misaligned_model, vgg_perpixel_model, G_T_optimizer, D_clear_optimizer, D_adv_optimizer, \
    clear_train_dataset, fog_train_dataset, snow_train_dataset, weather_type, args):

    clear_real_now, clear_ref_now, adv_syn_now = next(clear_train_dataset)
    beta_idx = random.randint(args.beta_min,args.beta_max)
    adv_syn_now = adv_syn_now[:,:,:,beta_idx*3:(beta_idx + 1)*3]  

    if weather_type == 'fog':
        adv_real_l, adv_real_r, adv_warped_l, adv_ref, adv_gated, adv_disp, adv_depth = next(fog_train_dataset)
        gated_supervision_weight = 0
    elif weather_type == 'snow':
        adv_real_l, adv_real_r, adv_warped_l, adv_ref, adv_gated, adv_disp, adv_depth = next(snow_train_dataset)
    else:
        assert False, "Undefined weather type."

    with tf.GradientTape(persistent=True) as tape:
        clear_fake_syn_now = G_T(adv_syn_now, training = True)
        clear_same_now = G_T(clear_real_now, training = True)

        disc_clear_real = D_clear(clear_ref_now, training = True)

        clear_fake_real_now = G_T(adv_real_l, training = True)
        if weather_type == 'fog':
            disc_adv_real = D_adv(adv_ref, training = True)
            adv_fake = Foggify(clear_fake_real_now, adv_ref, adv_depth)
            disc_adv_fake = D_adv(adv_fake, training = True)
        disc_clear_fake = D_clear(tf.concat([clear_fake_real_now, clear_fake_syn_now],axis = 0), training = True)
        tv_loss = TV_reg_loss(tf.concat([clear_fake_real_now, clear_fake_syn_now],axis = 0), args.tv_loss_weight)


        gen_c_loss = generator_loss(disc_clear_fake,args. adv_loss_weight) 
        if weather_type == 'fog':
            gen_c_loss = gen_c_loss + generator_loss(disc_adv_fake, 0.1*args.adv_loss_weight)
        paired_loss_b2c = paired_supervision_loss(clear_fake_syn_now, clear_ref_now, vgg_perpixel_model, args.perc_loss_weight, args.spatial_loss_weight, args.paired_loss_weight) 
        identity_loss_c = paired_supervision_loss(clear_same_now, clear_ref_now, vgg_perpixel_model, args.perc_loss_weight, args.spatial_loss_weight, args.identity_loss_weight) 
        total_G_T_loss = gen_c_loss +  paired_loss_b2c + identity_loss_c + tv_loss

        if args.gated_supervision_weight > 0:
            gated_loss = gated_supervision_loss(clear_fake_real_now, adv_gated, RGB2Gated, vgg_misaligned_model, args.gated_supervision_weight)
            total_G_T_loss = total_G_T_loss + gated_loss

        disc_c_loss = discriminator_loss(disc_clear_real, disc_clear_fake)
        if weather_type == 'fog':
            disc_b_loss = discriminator_loss(disc_adv_real, disc_adv_fake)
     
    G_T_gradients = tape.gradient(total_G_T_loss, G_T.trainable_variables)
    G_T_optimizer.apply_gradients(zip(G_T_gradients, G_T.trainable_variables))

    D_clear_gradients = tape.gradient(disc_c_loss, D_clear.trainable_variables)
    D_clear_optimizer.apply_gradients(zip(D_clear_gradients, D_clear.trainable_variables)) 
    if weather_type == 'fog':
        D_adv_gradients = tape.gradient(disc_b_loss, D_adv.trainable_variables)
        D_adv_optimizer.apply_gradients(zip(D_adv_gradients, D_adv.trainable_variables))    

def train_consistency_step(G_T, G_C, vgg_perpixel_model, G_C_optimizer, fog_train_dataset, snow_train_dataset, weather_type,args):
    if weather_type == 'fog':
        adv_real_l, adv_real_r, adv_warped_l, adv_ref, adv_gated, adv_disp, adv_depth = next(fog_train_dataset)
        gated_supervision_weight = 0
    elif weather_type == 'snow':
        adv_real_l, adv_real_r, adv_warped_l, adv_ref, adv_gated, adv_disp, adv_depth = next(snow_train_dataset)
    else:
        assert False, "Undefined weather type."

    clear_fake_real_now = G_T(adv_real_l, training = False)

    if args.temporal_loss_weight > 0:
        clear_fake_warped_now = G_T(adv_warped_l, training = False)

    if args.stereo_loss_weight > 0:
        in_r_warped = left2right(adv_real_r, -1 * adv_disp)
        out_r_warped = G_T(in_r_warped, training = False)

    with tf.GradientTape(persistent=True) as tape:
        clear_fake_real_now_enhanced = G_C(clear_fake_real_now, training = True)
        if args.temporal_loss_weight > 0:
            temporal_loss = paired_supervision_loss(clear_fake_real_now_enhanced, clear_fake_warped_now, vgg_perpixel_model, args.perc_loss_weight, args.spatial_loss_weight, args.temporal_loss_weight) 
        else:
            temporal_loss = 0
        if args.stereo_loss_weight > 0:
            out_r_enhanced = G_C(out_r_warped, training = True)
            stereo_consistency_loss = stereo_loss(clear_fake_real_now_enhanced, out_r_enhanced, adv_real_l, args.stereo_loss_weight)
        else:
            stereo_consistency_loss = 0


        total_loss = temporal_loss + stereo_consistency_loss

    G_C_gradients = tape.gradient(total_loss, G_C.trainable_variables)
    G_C_optimizer.apply_gradients(zip(G_C_gradients, G_C.trainable_variables))    

def train(args):

    ## Initialization
    G_T = build_model(args.G_T, args.in_h, args.in_w)
    D_clear = build_model(args.D_clear, args.in_h, args.in_w)
    D_adv = build_model(args.D_adv, args.in_h, args.in_w)
    G_C = build_model(args.G_C, args.in_h, args.in_w)

    G_T_optimizer = tf.keras.optimizers.Adam(args.Gt_lr, beta_1=0.9)
    D_clear_optimizer = tf.keras.optimizers.Adam(args.D_lr, beta_1=0.9)
    D_adv_optimizer = tf.keras.optimizers.Adam(args.D_lr, beta_1=0.9)
    G_C_optimizer = tf.keras.optimizers.Adam(args.Gc_lr, beta_1=0.9)

    RGB2Gated = tf.keras.models.load_model(args.rgb2gated_network)

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    vgg_perpixel_model = tf.keras.Model(inputs=vgg.input, outputs=[vgg.get_layer(name).output for name in args.vgg_perpixel_layers.split(',')])
    vgg_misaligned_model = tf.keras.Model(inputs=vgg.input, outputs=[vgg.get_layer(name).output for name in args.vgg_misaligned_layers.split(',')])

    checkpoint = tf.train.Checkpoint(
        G_T = G_T, G_T_optimizer = G_T_optimizer, 
        D_clear = D_clear, D_clear_optimizer = D_clear_optimizer, 
        D_adv = D_adv, D_adv_optimizer = D_adv_optimizer, 
        G_C = G_C, G_C_optimizer = G_C_optimizer)
    manager = tf.train.CheckpointManager(checkpoint, directory=args.result_path, max_to_keep=10)
    summary_writer = tf.summary.create_file_writer(args.result_path)

    ## Load data
    clear_train_dataset = load_clear_tfrecord(args.train_clear_tfrecord, args.load_h, args.load_w, args.in_h, args.in_w)
    fog_train_dataset = load_warped_tfrecord(args.train_fog_tfrecord, args.load_h, args.load_w, args.in_h, args.in_w)
    snow_train_dataset = load_warped_tfrecord(args.train_snow_tfrecord, args.load_h, args.load_w, args.in_h, args.in_w)

    clear_val_dataset = load_clear_tfrecord(args.val_clear_tfrecord, args.load_h, args.load_w, args.load_h, args.load_w, False)
    fog_val_dataset = load_warped_tfrecord(args.val_fog_tfrecord, args.load_h, args.load_w, args.load_h, args.load_w, False)
    snow_val_dataset = load_sequence_tfrecord(args.val_snow_tfrecord, args.load_h, args.load_w, args.load_h, args.load_w, False)
    
    clear_val = []
    fog_val = []
    snow_val = []

    for i in range(args.test_size):
        clear_val.append(next(clear_val_dataset))
        fog_val.append(next(fog_val_dataset))
        snow_val.append(next(snow_val_dataset))

    ## Training
    for step in range(args.n_steps):  
        ## Logging and saving
        if step % args.log_freq == 0:
            log(G_T, D_clear, D_adv, G_C, RGB2Gated, vgg_misaligned_model, vgg_perpixel_model,\
                clear_val[0], fog_val[0], summary_writer, step, args)

        if step % args.save_freq == 0:
            save_performance(G_T, G_C, clear_val, fog_val, snow_val, step, args)
            manager.save()
            if args.train_translation or step == 0:     
                G_T.save(os.path.join(args.result_path, "Translation_%02d" % (step/args.save_freq)))
                D_clear.save(os.path.join(args.result_path, "DiscClear_%02d" % (step/args.save_freq)))
                D_adv.save(os.path.join(args.result_path, "DiscAdv_%02d" % (step/args.save_freq)))
            if args.train_consistency:
                G_C.save(os.path.join(args.result_path, "Consistency_%02d" % (step/args.save_freq)))

        if args.weather_type == 'fog' or (args.weather_type == 'mixed' and step % 2 == 0):
            if args.train_translation:
                train_translation_step(G_T, D_clear, D_adv, RGB2Gated, vgg_misaligned_model, vgg_perpixel_model, G_T_optimizer, D_clear_optimizer, D_adv_optimizer,
                clear_train_dataset, fog_train_dataset, snow_train_dataset, 'fog', args)
            if args.train_consistency:
                train_consistency_step(G_T, G_C, vgg_perpixel_model, G_C_optimizer, fog_train_dataset, snow_train_dataset, 'fog',args)
        else:
            if args.train_translation:
                train_translation_step(G_T, D_clear, D_adv, RGB2Gated, vgg_misaligned_model, vgg_perpixel_model, G_T_optimizer, D_clear_optimizer, D_adv_optimizer,
                clear_train_dataset, fog_train_dataset, snow_train_dataset, 'snow', args)
            if args.train_consistency:
                train_consistency_step(G_T, G_C, vgg_perpixel_model, G_C_optimizer, fog_train_dataset, snow_train_dataset, 'snow',args)

def main():
    parser = argparse.ArgumentParser(
        description='Train ZeroScatter',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    def str2bool(v):
        assert(v == 'True' or v == 'False')
        return v.lower() in ('true')

    def none_or_str(value):
        if value.lower() == 'none':
            return None
        return value

# Data loading arguments
    parser.add_argument('--train_clear_tfrecord', required = True, type = str, help='Directory of clear weather training tfrecord')
    parser.add_argument('--train_fog_tfrecord', required = True, type = str, help='Directory of fog weather training tfrecord')
    parser.add_argument('--train_snow_tfrecord', required = True, type = str, help='Directory of snow weather training tfrecord')
    parser.add_argument('--val_clear_tfrecord', required = True, type = str, help='Directory of clear weather validation tfrecord')
    parser.add_argument('--val_fog_tfrecord', required = True, type = str, help='Directory of fog weather validation tfrecord')
    parser.add_argument('--val_snow_tfrecord', required = True, type = str, help='Directory of snow weather validation tfrecord')
    parser.add_argument('--load_h', default = 768, type=int, help='Image height when loading')
    parser.add_argument('--load_w', default = 1280, type=int, help='Image width when loading')
    parser.add_argument('--in_h', default = 512, type=int, help='Image height after processing')
    parser.add_argument('--in_w', default = 512, type=int, help='Image height after processing')

    # Saving and logging arguments
    parser.add_argument('--result_path', required = True, type = str, help='Directory for saving ckpts and TensorBoard file.')
    parser.add_argument('--test_size', default=2, type=int, help='Number of test images per weather type')
    parser.add_argument('--log_freq', default=500, type=int, help='Interval to write to TensorBoard')
    parser.add_argument('--save_freq', default=1000, type=int, help='Interval to save model')     

    # Training arguments
    parser.add_argument('--Gt_lr', default = 5e-5, type=float, help='Translation block learning rate')
    parser.add_argument('--D_lr', default = 5e-5, type=float, help='Discriminator learning rate')
    parser.add_argument('--Gc_lr', default = 5e-5, type=float, help='Consistency block learning rate')

    parser.add_argument('--G_T', default = 'translation', type=str, help='Translation block model')
    parser.add_argument('--D_clear', default = 'discriminator', type=str, help='Clear weather discriminator model')
    parser.add_argument('--D_adv', default = 'discriminator', type=str, help='Adverse weather discriminator model')
    parser.add_argument('--G_C', default = 'consistency', type=str, help='Consistency block model')

    parser.add_argument('--n_steps', default = 50000, type=int, help='Number of training steps')
    parser.add_argument('--train_translation', default=True, type=str2bool, help='Train translation block or not')
    parser.add_argument('--train_consistency', default=False, type=str2bool, help='Train consistency block or not')
    parser.add_argument('--weather_type', default = 'fog', type = str, help='Type of adverse weather, choose between fog, snow, or mixed')
    parser.add_argument('--beta_min', default = 0, type = int, help='Lightest fog intensity level')
    parser.add_argument('--beta_max', default = 6, type = int, help='Highest fog intensity level')

    # Loss arguments    
    parser.add_argument('--rgb2gated_network', default=None, type=none_or_str, help='Directory of pretrained RGB2Gated model')
    parser.add_argument('--vgg_perpixel_layers', default='block2_conv1,block3_conv1', type=str, help='Layers used in perceptual loss - lower level')
    parser.add_argument('--vgg_misaligned_layers', default='block3_conv1,block4_conv1,block5_conv1', type=str, help='Layers used in perceptual loss - higher level')

    parser.add_argument('--gated_supervision_weight', default=0.2, type=float, help='Gated supervision loss weight')
    parser.add_argument('--identity_loss_weight', default=5, type=float, help='Identity loss weight')
    parser.add_argument('--paired_loss_weight', default=5, type=float, help='Paired supervision loss weight')
    parser.add_argument('--adv_loss_weight', default=0.1, type=float, help='Adversarial loss weight')
    parser.add_argument('--perc_loss_weight', default=0.01, type=float, help='Perceptual loss weight')
    parser.add_argument('--spatial_loss_weight', default=1, type=float, help='Spatial loss weight')
    parser.add_argument('--tv_loss_weight', default=0, type=float, help='Total variation loss weight')
    parser.add_argument('--temporal_loss_weight', default=5, type=float, help='Temporal consistency loss weight')
    parser.add_argument('--stereo_loss_weight', default=5, type=float, help='Stereo consistency loss weight')

    args = parser.parse_args()
    
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    args_dict = vars(args)
    with open(os.path.join(args.result_path,'args.json'), "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=False)
        
    train(args)

if __name__ == '__main__':
    main()