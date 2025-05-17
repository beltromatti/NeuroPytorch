# NeuroPytorch (ntorch)

**NeuroPytorch (ntorch)** is a custom fork of PyTorch designed to support **low-level control** and **hardware-maximized operations** for experimenting with more **biologically inspired architectures**, with the ultimate goal of accelerating research towards **Artificial General Intelligence (AGI)**.

This version builds upon vanilla PyTorch by introducing both:
- custom operators and kernels to improve the overall performance,
- a foundation for future high-efficiency neural designs that are not yet supported by general-purpose frameworks like PyTorch or TensorFlow.

> NeuroPytorch is still in active development and will evolve in response to the author's research needs, particularly in exploring novel computation models inspired by the structure and function of the human brain.

---

## 🔍 Key Features

### ✅ FusedLinearReLU operator (CPU & CUDA)

A fully custom **fused operation** that combines matrix multiplication, bias addition, and ReLU activation into a single low-level kernel — designed for **maximum throughput and minimal memory traffic**.

#### 🧠 Purpose:
Improve performance by reducing kernel dispatches and memory reads/writes — especially during inference or in inner loops of large models.

#### ⚙️ Signature:

```python
y = torch.ops.aten.fused_linear_relu(x, w, b)
# equivalent to: y = relu(x @ w.T + b)
```

---

### 📂 Implementation Overview

#### 📁 CPU Implementation
Location: `aten/src/ATen/native/cpu/FusedLinearRelu.cpp`

- Pure C++ implementation with manual loop fusion.
- Uses `at::parallel_for` for thread-level parallelism.
- Leverages `AT_DISPATCH_FLOATING_TYPES` for float/double genericity.
- Can be further optimized using AVX intrinsics or cBLAS fallback.

#### 📁 CUDA Implementation
Location: `aten/src/ATen/native/cuda/FusedLinearRelu.cu`

- CUDA kernel that fuses all operations into one pass.
- Launches with thread blocks optimized for memory coalescing.
- Uses shared memory when possible.
- Compatible with TensorCore acceleration (FP16) in future versions.

#### 🧾 Dispatcher registration
Location: `aten/src/ATen/native/native_functions.yaml`

```yaml
- func: fused_linear_relu(Tensor input, Tensor weight, Tensor bias) -> Tensor
  variants: function
  dispatch:
    CPU: fused_linear_relu_cpu
    CUDA: fused_linear_relu_cuda
```

This allows PyTorch’s dispatcher to call the appropriate backend based on tensor device.

---

## 📜 License for Custom Extensions

All **custom extensions and optimizations added in NeuroPytorch** are released under the **GNU General Public License v3.0 (GPL-3.0)**.

The rest of PyTorch remains under its original BSD-style license. This dual-licensing reflects the original work of the PyTorch team as well as the AGI-oriented goals of this derivative work.

---

## 🚧 Roadmap & Research Direction

NeuroPytorch will continuously evolve to support novel research in **bio-inspired AI architectures**, with future features including:

- Event-driven, spiking-like computation layers;
- Local learning rule modules (Hebbian, STDP, predictive coding);
- Neuromorphic simulation kernels (CPU/GPU);
- Recurrent microcircuits with stateful dynamics;
- Differentiable approximations of cortical motifs;
- Native support for heterogeneous execution graphs (e.g., sparse + dense hybrid logic).

> This project is exploratory in nature and prioritizes **scientific experimentation** over general-purpose deployment. It is intended as a foundation for research into fundamentally new learning architectures.

---

## 🧪 Development Notes

To build from source after modifying C++/CUDA kernels:

```bash
python setup.py develop
```

Or for a full clean build:

```bash
rm -rf build
python setup.py develop
```

---

## 📫 Contact

This project is maintained by [Mattia Beltrami], a Computer Science student researching low-level efficiency and biological realism in machine learning. Feedback and collaboration are welcome.