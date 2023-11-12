# Integrated-Photonic-Encoder-for-Terapixel-Image-Processing  
Offical repository for "[Wang, Xiao, Brandon Redding, Nicholas Karl, Christopher Long, Zheyuan Zhu, Shuo Pang, David Brady, and Raktim Sarma. "Integrated Photonic Encoder for Terapixel Image Processing." arXiv preprint arXiv:2306.04554 (2023)](https://arxiv.org/abs/2306.04554)"  

## Jupyter Notebooks
[train_decoder.ipynb](https://github.com/djbradyAtOpticalSciencesArizona/Integrated-Photonic-Encoder-for-Terapixel-Image-Processing/blob/main/train_decoder.ipynb): Construction and training of a neural network optimized for a compressive ratio of 8:1. The network takes compressed data as input and outputs original images.  
[demo_decoder.ipynb](https://github.com/djbradyAtOpticalSciencesArizona/Integrated-Photonic-Encoder-for-Terapixel-Image-Processing/blob/main/demo_decoder.ipynb): A guide on using a pre-trained model to reconstruct the original images from their compressed versions.  

## Files in data/
### File formats
- Ground truth data
  - dimension: 512 $×$ 512 (H $\times$ W)
  - format: PNG
  - total number: 19
- Compressed data
  - dimension: 64 $×$ 64 $×$ 8 (H $\times$ W $\times$ C)
  - format: .npy
  - total number: 19

The corresponding compressed data of "000005_10.png" is "000005_10.npy", and the rest files follow the same pattern.
