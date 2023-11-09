# Integrated-Photonic-Encoder-for-Terapixel-Image-Processing  
Offical repository for "[Wang, Xiao, Brandon Redding, Nicholas Karl, Christopher Long, Zheyuan Zhu, Shuo Pang, David Brady, and Raktim Sarma. "Integrated Photonic Encoder for Terapixel Image Processing." arXiv preprint arXiv:2306.04554 (2023)](https://arxiv.org/abs/2306.04554)"  

The notebook "[train_decoder.ipynb](https://github.com/djbradyAtOpticalSciencesArizona/Integrated-Photonic-Encoder-for-Terapixel-Image-Processing/blob/main/train_decoder.ipynb)" demonstrates the construction and training of a neural network optimized for a compressive ratio of 8:1. The network takes 64 inputs and produces 8 outputs.  
The notebook "demo_decoder.ipynb" provides a guide on using a pre-trained model to reconstruct the original images from their compressed versions.  

## file formats in data/  
- Ground truth data
  - dimension: 512 $×$ 512 (H $\times$ W)
  - format: PNG
- Compressed data
  - dimension: 64 $×$ 64 $×$ 8 (H $\times$ W $\times$ C)
  - format: .npy
