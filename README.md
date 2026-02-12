# 📡 ML Autoencoder for Communication Systems

This project implements a neural network-based autoencoder to simulate and enhance digital communication over a noisy channel using machine learning techniques. It integrates concepts from digital modulation and deep learning.

---

## 🚀 Project Overview

The system focuses on transmitting image data across a noisy channel using:

* **QPSK (Quadrature Phase-Shift Keying) modulation/demodulation**
* **AWGN (Additive White Gaussian Noise) channel modeling**
* **Autoencoder architecture** for compression, noise mitigation, and reconstruction

The pipeline includes:

1. Encoding image data into a compressed representation (7×7×8 tensor)
2. Converting to a 1×392 bitstream
3. Modulating and passing through a noisy channel
4. Demodulating and decoding to reconstruct the image

As shown on *page 5 of the project PDF*, this structure helps simulate realistic communication system behavior while exploring the robustness of deep learning to noise【41†Institutsprojekt - ICE.pdf】.

---

## 🎯 Goals

* Combine classical communication techniques with neural networks
* Understand autoencoder behavior in noisy environments
* Experiment with varying SNR (Signal-to-Noise Ratio) and observe output fidelity【41†Institutsprojekt - ICE.pdf】

---

## 🔧 Framework Versions

| Package        | Version | Notes                                                       |
| -------------- | ------- | ----------------------------------------------------------- |
| **NumPy**      | 1.25.2  | Required by `komm`; avoids deprecated `np.complex` removal. |
| **Scipy**      | 1.10.1  | Compatible with TensorFlow 2.13.0 and NumPy 1.25.2.         |
| **TensorFlow** | 2.13.0  | Fully tested with NumPy 1.25.x.                             |
| **Keras**      | 2.13.1  | Matches TensorFlow 2.13.0 version family.                   |
| **Komm**       | 0.7.0   | Requires NumPy < 1.26 due to legacy type usage.             |

---

## 📥 Installation

Use the following commands in a clean environment:

```bash
pip uninstall tensorflow keras komm scipy numpy -y
pip cache purge

pip install numpy==1.25.2
pip install scipy==1.10.1
pip install tensorflow==2.13.0
pip install keras==2.13.1
pip install komm==0.7.0
```

---

## ✅ Test Setup

```python
import numpy as np
import scipy
import tensorflow as tf
import keras
import komm

print("NumPy:", np.__version__)
print("Scipy:", scipy.__version__)
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("Komm:", komm.__version__)

QPSK = komm.PSKModulation(4, phase_offset=0)
print("QPSK modulation initialized:", QPSK)
```

---

## 📁 Source Code

See `ML Autoencoder.py` for the complete implementation of the autoencoder, including:

* Model architecture
* Training pipeline
* QPSK channel simulation

---

## 📌 Author

This is a personal project developed to explore the integration of deep learning with digital communication systems.
