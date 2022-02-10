import os
import argparse
import json
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(2)
from matplotlib import pyplot
from utils import *
from loss import *


def save_performance(G, val, step, args):
    pyplot.figure(figsize=(20,20))
    for i in range(len(val)):
        img, gt_img = val[i]
        G_img = G(img, training=False)

        mask = ent_mask(gt_img)
        masked_fake_gated = tf.where(mask, G_img, tf.zeros_like(G_img))
        masked_true_gated = tf.where(mask, gt_img, tf.zeros_like(gt_img))

        pyplot.subplot(args.test_size, 3, 3*i+1)
        pyplot.axis('off')
        if i == 0:
            pyplot.title('Input', fontdict={'fontsize':20})
        pyplot.imshow((tf.squeeze(img).numpy()))        
        pyplot.subplot(args.test_size, 3, 3*i+2)
        pyplot.axis('off')
        if i == 0:
            pyplot.title('Output', fontdict={'fontsize':20})
        pyplot.imshow(tf.squeeze(masked_fake_gated))
        pyplot.subplot(args.test_size, 3, 3*i+3)
        pyplot.axis('off')
        if i == 0:
            pyplot.title('GT', fontdict={'fontsize':20})
        pyplot.imshow(tf.squeeze(masked_true_gated))    

    # save plot to file
    filename = os.path.join(args.result_path,'plot_%02d.png' % (step/args.save_freq))
    pyplot.savefig(filename)
    pyplot.close()

def log(G, D, vgg_model, val, summary_writer, step, args):
    input_img, gt_img = val
    G_img = G(input_img, training=False)
    D_real = D([input_img, gt_img], training=False)
    D_fake = D([input_img, G_img ], training=False)

    perc_loss = P_loss(G_img, gt_img, vgg_model, args.perc_loss_weight)
    edge_loss = Edge_loss(G_img, gt_img, args.edge_loss_weight)
    adv_loss = generator_loss(D_fake,args.adv_loss_weight)
    G_loss = perc_loss + edge_loss + adv_loss
    D_loss = discriminator_loss(D_real, D_fake)
    with summary_writer.as_default():
        tf.summary.scalar(name = 'loss/adv_loss', data = adv_loss, step=step)
        tf.summary.scalar(name = 'loss/perc_loss', data = perc_loss, step=step)
        tf.summary.scalar(name = 'loss/edge_loss', data = edge_loss, step=step)
        tf.summary.scalar(name = 'loss/G_loss'    , data = G_loss   , step=step)
        tf.summary.scalar(name = 'loss/D_loss'    , data = D_loss   , step=step)

def train_step(G, D, vgg_model, G_optimizer, D_optimizer, train_dataset, args):
    input_img, gt_img = next(train_dataset)
    if tf.reduce_mean(gt_img) < 0.6: # filter out over-exposed gated images
        with tf.GradientTape(persistent=True) as tape:
            G_img = G(input_img, training=True)
            D_real = D([input_img, gt_img], training=True)
            D_fake = D([input_img, G_img ], training=True)

            perc_loss = P_loss(G_img, gt_img, vgg_model, args.perc_loss_weight)
            edge_loss = Edge_loss(G_img, gt_img, args.edge_loss_weight)
            adv_loss = generator_loss(D_fake,args.adv_loss_weight)
            G_loss = perc_loss + edge_loss + adv_loss
            D_loss = discriminator_loss(D_real, D_fake)

        
        G_gradients = tape.gradient(G_loss, G.trainable_variables)
        G_optimizer.apply_gradients(zip(G_gradients, G.trainable_variables))
        D_gradients = tape.gradient(D_loss, D.trainable_variables)
        D_optimizer.apply_gradients(zip(D_gradients, D.trainable_variables)) 

def train(args):

    ## Initialization
    G = build_model(args.G, args.in_h, args.in_w)
    D = build_model(args.D, args.in_h, args.in_w)

    G_optimizer = tf.keras.optimizers.Adam(args.G_lr, beta_1=0.9)
    D_optimizer = tf.keras.optimizers.Adam(args.D_lr, beta_1=0.9)

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    vgg_model = tf.keras.Model(inputs=vgg.input, outputs=[vgg.get_layer(name).output for name in args.vgg_layers.split(',')])

    checkpoint = tf.train.Checkpoint(G = G, G_optimizer = G_optimizer, D= D, D_optimizer = D_optimizer)
    manager = tf.train.CheckpointManager(checkpoint, directory=args.result_path, max_to_keep=10)
    summary_writer = tf.summary.create_file_writer(args.result_path)

    ## Load data
    train_dataset = load_rgb2gated_tfrecord(args.train_tfrecord, args.load_h, args.load_w, args.in_h, args.in_w)
    val_dataset = load_rgb2gated_tfrecord(args.val_tfrecord, args.load_h, args.load_w, args.load_h, args.load_w, False)
    
    val = []
    for i in range(args.test_size):
        val.append(next(val_dataset))

    ## Training
    for step in range(args.n_steps):  
        ## Logging and saving
        if step % args.log_freq == 0:
            log(G, D, vgg_model, val[0], summary_writer, step, args)

        if step % args.save_freq == 0:
            save_performance(G, val, step, args)
            manager.save()
            G.save(os.path.join(args.result_path, "RGB2Gated_%02d" % (step/args.save_freq)))

        train_step(G, D, vgg_model, G_optimizer, D_optimizer, train_dataset, args)


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
    parser.add_argument('--train_tfrecord', required = True, type = str, help='Directory of training tfrecord')
    parser.add_argument('--val_tfrecord', required = True, type = str, help='Directory of validation tfrecord')
    parser.add_argument('--load_h', default = 512, type=int, help='Image height when loading')
    parser.add_argument('--load_w', default = 1024, type=int, help='Image width when loading')
    parser.add_argument('--in_h', default = 512, type=int, help='Image height after processing')
    parser.add_argument('--in_w', default = 1024, type=int, help='Image height after processing')

    # Saving and logging arguments
    parser.add_argument('--result_path', required = True, type = str, help='Directory for saving ckpts and TensorBoard file.')
    parser.add_argument('--test_size', default=3, type=int, help='Number of test images')
    parser.add_argument('--log_freq', default=500, type=int, help='Interval to write to TensorBoard')
    parser.add_argument('--save_freq', default=1000, type=int, help='Interval to save model')    

    # Training arguments
    parser.add_argument('--G_lr', default = 1e-4, type=float, help='Generator learning rate')
    parser.add_argument('--D_lr', default = 1e-4, type=float, help='Discriminator learning rate')
    parser.add_argument('--G', default = 'rgb2gated', type=str, help='Generator model')
    parser.add_argument('--D', default = 'rgb2gated_disc', type=str, help='Discriminator model')
    parser.add_argument('--n_steps', default = 40000, type=int, help='Number of training steps')

    # Loss arguments    
    parser.add_argument('--vgg_layers', default='block4_conv2,block5_conv2', type=str, help='Layers used in perceptual loss')
    parser.add_argument('--adv_loss_weight', default=0.01, type=float, help='Adversarial loss weight')
    parser.add_argument('--perc_loss_weight', default=0.01, type=float, help='Perceptual loss weight')
    parser.add_argument('--edge_loss_weight', default=10, type=float, help='Edge loss weight')

    args = parser.parse_args()
    
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    args_dict = vars(args)
    with open(os.path.join(args.result_path,'args.json'), "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=False)
        
    train(args)

if __name__ == '__main__':
    main()