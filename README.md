# Seq2Seq-Pytorch
Implementation of famous Seq2Seq Papers using Pytorch
## Goal
Implementing a model for machine translation from German to English using the concepts decribed in the follwing research papers.
- [x] <a href="https://arxiv.org/abs/1409.3215">Sequence to Sequence Learning with Neural Networks</a>
- [x] <a href="https://arxiv.org/abs/1406.1078">Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation</a>
- [ ] <a href="https://arxiv.org/abs/1705.03122">Convolutional Sequence to Sequence Learning</a>
- [ ] <a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a>
## Getting Started
Install pytorch to the latest version(with or without CUDA compatibility depending on your system configuration). Use the link below for detailed installation process.<br><a href="https://pytorch.org/">Pytorch-Install</a><br><br>
Install torchtext using the follwoing command:<br>
```pip3 install torchtext```<br><br>
We will be using spacy to tokenize our input and target languages. In this tutorial we are going to use German(Input) & English(Target) as our languages. For that, execute the follwoing commands:<br>
```python -m spacy download en```<br>
```python -m spacy download de```<br><br>
Check <a href="https://spacy.io/usage">Here</a> for Spacy Installation instructions<br>

### If you find any mistakes , please do not hesitate to submit an issue. I welcome any feedback, positive or negative!
