# Sparse-recovery-of-IR-images-using-a-score-based-model
Sparse recovery of IR images using a score-based model

# There are multiple python files in this directory for different purposes
- square_IR_main.py
Training on the FLIR dataset and reconstruction (generation out of noise using ODE sampler is also supported)

- mnist_main.py
Training on the MNIST dataset and reconstruction (generation out of noise using ODE sampler is also supported)

- mnist_ncsn_main.py (Work in progress still, sadly...)
Training on the MNIST dataset and using Yang Songs ResNet implemnetation for the score work and reconstruction

To train a score network go into _main.py files and change task to 'train', To generate 'generate' and to reconstruct change it to 'reconstruct'

"process_data_set.py" processes the images in the images/data DIR according to a certain split and downsampling ratio

"analyse_sparsity.py" Is used to find the amount of non-zero components that can be used for sparse recovery

Convolutional network size calculator.xlsx was used to design the layers of the CNN for the scorenet.