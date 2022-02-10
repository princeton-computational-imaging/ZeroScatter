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

WEATHER_TYPE=snow

TRAIN_TRANSLATION=False
TRAIN_CONSISTENCY=True
N_STEPS=20000

G_T=[Directory of pretrained Translation Block]
D_CLEAR=[Directory of pretrained discriminator for clear weather]
D_ADV=[Directory of pretrained discriminator for adverse weather]

conda activate zeroscatter
rm -r $RESULT_DIR
python train.py --train_clear_tfrecord $TRAIN_CLEAR_TFRECORD --train_fog_tfrecord $TRAIN_FOG_TFRECORD --train_snow_tfrecord $TRAIN_SNOW_TFRECORD --val_clear_tfrecord $VAL_CLEAR_TFRECORD --val_fog_tfrecord $VAL_FOG_TFRECORD --val_snow_tfrecord $VAL_SNOW_TFRECORD --train_consistency $TRAIN_CONSISTENCY --train_translation $TRAIN_TRANSLATION --n_steps $N_STEPS --result_path $RESULT_DIR --weather_type $WEATHER_TYPE --rgb2gated_network $RGB2GATED --G_T $G_T --D_clear $D_CLEAR --D_adv $D_ADV