# ResNet-for-JAX

[![Project Status: Not done](https://img.shields.io/badge/status-in_progress-orange.svg)](https://tugdual.fr)

A JAX-based implementation of the Perciever using the Equinox framework. This model is part of the JAXTTS (eXtended Text-To-Speech) series, where I rewrite XTTS in JAX to understand how it works from A to Z, and learn JAX along the way.

## Overview

This project leverages **JAX** and **Equinox** for a Perciever model.

## Features

- A convolutional network mapping Mel Spectrograms to a latent representation of 512 dim vectors, that can then be fed to HiFiGAN. 
- Comprehensive JAX and Equinox integration
- Documentation with step-by-step tutorials and explanations for each module
- A notebook with code to map the weights of the Coqui-ai's to the JAX implementation. Outputs have very high similarity with the model weights of XTTS.

## Getting Started

To get started, clone the repository and follow the tutorial on https://tugdual.fr/ResNet-for-JAX/# perciever-jax
