# 🌼 Xuyến Chi (Bidens Pilosa)

<div align="center">

![Xuyen Chi Flower](https://images.unsplash.com/photo-1628628043644-8d45ec6ba94d?q=80&w=2574&auto=format&fit=crop)

*"Beauty lies in resilience. Intelligence lies in efficiency."*

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)]()
[![Model Type](https://img.shields.io/badge/Model-Linear%20State%20Space-blue.svg)]()
[![Optimizer](https://img.shields.io/badge/Optimizer-Pilosa-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Blooming-purple.svg)]()

</div>

---

## 📜 The Philosophy: Why "Xuyến Chi"?

In the lush landscapes of Vietnam, **Xuyến Chi** (Bidens pilosa) is a wildflower known for its incredible resilience. It grows where others cannot, its seeds hitchhiking on passersby to spread life across vast distances.

> *"Hoa dại ven đường không ai cấy*  
> *Mặc kệ nắng mưa vẫn trổ bông"*
>
> **"Wildflowers by the road, planted by no one,**  
> **Defying sun and rain, they still bloom."**

This model mimics that spirit. Unlike the heavy, resource-hungry giants of the AI world, **Xuyến Chi** is designed to be lightweight, adaptive, and enduring. It utilizes a **Linear State Space** architecture—allowing it to "hitchhike" on limited hardware while covering long-range dependencies, just as the flower covers the land.

---

## 🧬 Anatomy of the Architecture

The **Xuyến Chi** architecture is a poetic composition of three organic components, each meticulously crafted to balance the flow of information.

### 1. The XylemMixer (The Heart) 🌿
Just as the *xylem* tissue in plants transports water and nutrients from roots to leaves, the **XylemMixer** transports gradients and context through time.

*   **Local Convolution (The Roots):** A 1D Convolution (`kernel=4`) captures immediate, local context. Like roots gripping the soil, this ensures the model understands the "now" before looking at the "forever."
*   **The Decay Mechanism (The Stem):**
    $$ \log(\text{decay}) = -8.0 \cdot (1 - \sigma(\text{gate})) - 0.1 $$
    This formula is the biological clock of the model. It determines what to remember and what to forget. The specific constants ($-8.0, -0.1$) act as evolutionary constraints, forcing the model to prioritize information efficiency.
*   **Parallel RNN Scan (The Sap Flow):**
    Instead of standard attention ($O(N^2)$), Xuyến Chi uses a cumulative scan ($O(N)$).
    $$ y_t = (e^{\text{decay}_t} \odot y_{t-1}) + x_t $$
    This allows the model to process infinite sequences in inference with a fixed memory footprint.

### 2. The Pilosa Optimizer (The Nutrient) 🧪
An optimizer specifically bred for this architecture.
*   **Gradient Centralization (GC):** `grad.sub_(grad.mean(...))`
    This technique stabilizes training by constraining the gradient vectors to lie on a hyperplane with zero mean. It creates a smoother loss landscape, allowing the model to converge like a flower opening at dawn—steady and consistent.

### 3. Structural Elegance 🏛️
*   **RMSNorm:** Ensures signal magnitude stability.
*   **SwiGLU:** A gated feed-forward network acting as the photosynthesis engine, converting raw representations into higher-order features.

---

## 💎 The "Money" Mechanism: What makes it unique?

While the world chases Transformers, Xuyến Chi introduces the **Hybrid Decay Gate**.

Most Linear RNNs use simple exponential decay. Xuyến Chi uses a **Learnable Parametric Decay** injected into the `Parallel Scan`.

```python
# The "Secret Sauce"
decay_gate = torch.sigmoid(base_decay + w_in)
log_decay = -8.0 * (1.0 - decay_gate) - 0.1
```

**Why this wins:**
1.  **Differentiable Forgetting:** The model *learns* the precise half-life of every feature.
2.  **Numerical Stability:** The formula prevents exploding gradients naturally without aggressive clipping.
3.  **The "Sticky" State:** Like the seeds of the Bidens pilosa, the hidden state ($h_t$) sticks to the memory only as long as necessary, then drops off to make room for new information.

---

## ⚔️ Clash of the Titans: Xuyến Chi vs. The Rest

| Feature | 🤖 Transformer | 🐍 Mamba (SSM) | 🌊 LSTM | 🌿 **Xuyến Chi** |
| :--- | :--- | :--- | :--- | :--- |
| **Complexity** | $O(T^2)$ (Quadratic) | $O(T)$ (Linear) | $O(T)$ (Linear) | **$O(T)$ (Linear)** |
| **Inference Speed**| Slow (KV Cache grows) | Fast (Fixed state) | Fast (Fixed state) | **Lightning (Parallel Scan)** |
| **Long Context** | Excellent but heavy | Excellent | Poor (Forget gate issues) | **Infinite (Decay Logic)** |
| **Parallel Training**| Yes | Yes | No (Sequential) | **Yes (Prefix Scan)** |
| **Philosophy** | Brute Force | Structured State | Gated Recurrence | **Organic Hybrid** |

*   **Vs. Transformer:** Xuyến Chi does not suffer from quadratic complexity. It can read a book of infinite length without running out of RAM.
*   **Vs. RWKV/Mamba:** Xuyến Chi simplifies the state transition matrix into a **Data-Dependent Decay Gate**, offering a more stable training path (thanks to `Pilosa`) and interpreted "forgetting" mechanics resembling biological neurons.

---

## 🚀 Quick Start

To plant your own Xuyến Chi garden:

```python
import torch
from xuyen_chi import XuyenChi, Pilosa

# 1. Initialize the seed
device = "cuda" if torch.cuda.is_available() else "cpu"
model = XuyenChi(vocab_size=10000, dim=512, n_layers=6, n_heads=8).to(device)

# 2. Prepare the nutrients (Optimizer)
optimizer = Pilosa(model.parameters(), lr=6e-4, use_gc=True)

# 3. Bloom (Generate)
input_ids = torch.tensor([[1, 2, 3]]).to(device)
output = model.generate(input_ids, max_new_tokens=100)
```

---

## 🎨 Code as Art

Every line of code in `XylemMixer` was written with rhythm.

> *The tensor reshapes are the breathing pattern.*
> *The GroupNorm is the skin.*
> *The Parallel Scan is the heartbeat.*

We believe that code should not just function; it should feel alive.

---

<div align="center">

*"Tuy là cỏ dại, vẫn mang cốt cách thanh cao."*
*(Though a weed, it carries a noble spirit.)*

**Made with ❤️ and ☕ in Vietnam.**

</div>
