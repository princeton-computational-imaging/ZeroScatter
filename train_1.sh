# Data loading
TRAIN_CLEAR_TFRECORD=[Directory of clear weather training tfrecord]
VAL_CLEAR_TFRECORD=[Directory of clear weather validation tfrecord]

TRAIN_FOG_TFRECORD=[Directory of foggy weather training tfrecord]
VAL_FOG_TFRECORD=[Directory of foggy weather validation tfrecord]

TRAIN_SNOW_TFRECORD=[Directory of snow weather training tfrecord]
VAL_SNOW_TFRECORD=[Directory of snow weather validation tfrecord]

RGB2GATED=[Directory of pretrained RGB2Gated model]

# Saving and logging
RESULT_DIR=[Directory for saving ckpts, tensorboard, etc]

WEATHER_TYPE=mix

TRAIN_TRANSLATION=True
TRAIN_CONSISTENCY=False

conda activate zeroscatter
rm -r $RESULT_DIR
python train.py --train_clear_tfrecord $TRAIN_CLEAR_TFRECORD --train_fog_tfrecord $TRAIN_FOG_TFRECORD --train_snow_tfrecord $TRAIN_SNOW_TFRECORD --val_clear_tfrecord $VAL_CLEAR_TFRECORD --val_fog_tfrecord $VAL_FOG_TFRECORD --val_snow_tfrecord $VAL_SNOW_TFRECORD --train_consistency $TRAIN_CONSISTENCY --train_translation $TRAIN_TRANSLATION --result_path $RESULT_DIR --weather_type $WEATHER_TYPE --rgb2gated_network $RGB2GATED