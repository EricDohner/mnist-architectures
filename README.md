# mnist-architectures
This is a personal ML experimentation project focused on MNIST digit classification using PyTorch. The main goal is exploring and comparing different neural network architectures while learning about gradient flow and attention mechanisms.

MNIST_classifier.py — Main entry point. Loads/preprocesses MNIST, trains a Vision Transformer model, and produces training visualizations including gradient flow plots.

transformer.py — A custom Vision Transformer (ViT) implementation. Written largely from scratch in order to zoom in and understand details.

networks.py — A collection of alternative architectures (MLP, CNN, Residual CNN) plus reusable training/testing loop utilities. Tracks batch loss, validation loss, and accuracy at regular intervals.

viz.py — Minimal visualization module. Plots training/validation curves and gradient norms across network layers.
