# ZeroScatter: Domain Transfer for Long Distance Imaging and Vision through Scattering Media
### [Project Page](https://light.princeton.edu/publication/zeroscatter/) | [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Shi_ZeroScatter_Domain_Transfer_for_Long_Distance_Imaging_and_Vision_Through_CVPR_2021_paper.html) | [Pretrained ckpts](https://drive.google.com/drive/folders/1Pv4n_pj8ZmBWtWcgmpuZyUqFBIcJy6sV?usp=sharing)

[Zheng Shi](https://zheng-shi.github.io/), [Ethan Tseng](https://ethan-tseng.github.io), [Mario Bijelic](http://mariobijelic.de/wordpress/), [Werner Ritter](), [Felix Heide](https://www.cs.princeton.edu/~fheide/)

If you find our work useful in your research, please cite:
```
@article{shi2021zeroscatter,
title={ZeroScatter: Domain Transfer for Long Distance Imaging and Vision through Scattering Media},
author={Shi, Zheng and Tseng, Ethan and Bijelic, Mario and Ritter, Werner and Heide, Felix},
journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2021}
}
```

## Requirements
This code is developed using TesnorFlow 2.2.0 on Linux machine. Full frozen environment can be found in 'environment.yml', note some of these libraries are not necessary to run this code. 

## Data
This code requires data in TFRecord format. If you are not familiar with it, you may find this [Tensorflow tutorial](https://www.tensorflow.org/tutorials/load_data/tfrecord#write_the_tfrecord_file) helpful. Please refer to the paper and the data loading functions in 'utils.py' for more detailed requirements for each training step.

## Testing
To perform inference on real-world captures, please first download the pre-trained model from [here](https://drive.google.com/drive/folders/1Pv4n_pj8ZmBWtWcgmpuZyUqFBIcJy6sV?usp=sharing) to 'ckpts/' folder, then you can run the 'inference.ipynb' notebook in Jupyter Notebook. The notebook will load the checkpoint and process captured sensor measurements located in 'tfrecord_example/'. The reconstructed images will be displayed within the notebook.

## Training
We include 3 bash scripts for training purpose:
- train_0.sh: performs training of RGB2Gated model, which is later used for indirect supervision based on gated captures.
- train_1.sh: performs training of the ZeroScatter translation block.
- train_2.sh: performs training of the ZeroScatter consistency block.

## License
Our code is licensed under BSL-1. By downloading the software, you agree to the terms of this License. The data in the folder 'tfrecord_example/' based on the [DENSE Dataset](https://www.uni-ulm.de/en/in/driveu/projects/dense-datasets/).

## Questions
If there is anything unclear, please feel free to reach out to me at zhengshi[at]princeton[dot]edu.