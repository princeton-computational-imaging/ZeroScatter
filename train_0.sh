# Data loading
TRAIN_TFRECORD=[Directory of training tfrecord]
VAL_TFRECORD=[Directory of training tfrecord]

# Saving and logging
RESULT_DIR=[Directory for saving ckpts, tensorboard, etc]

conda activate zeroscatter
rm -r $RESULT_DIR
python train_rgb2gated.py --train_tfrecord $TRAIN_TFRECORD --val_tfrecord $VAL_TFRECORD --result_path $RESULT_DIR