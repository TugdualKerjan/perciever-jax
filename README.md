# Perceiver-for-JAX

[![Project Status: Not done](https://img.shields.io/badge/status-in_progress-orange.svg)](https://tugdual.fr)

A JAX-based implementation of the Perceiver using the Equinox framework. This model is part of the JAXTTS (eXtended Text-To-Speech) series, where I rewrite XTTS in JAX to understand how it works from A to Z, and learn JAX along the way.

## Overview

This project leverages **JAX** and **Equinox** for a Perceiver model.

## 🚙 Roadmap

- [x] Have a functioning VAE
- [x] Provide checkpoints for the models
- [x] Documentation with step-by-step tutorials and explanations for each module

## Getting Started

To get started, follow the commands below. I recommend you use UV as a package manager:

```bash
git clone git@github.com:TugdualKerjan/audio-vae-jax.git
cd audio-vae-jax
uv sync
uv add jax["cuda"] # JAX has various versions optimized for the underlying architecture
uv run train_vae.py
```