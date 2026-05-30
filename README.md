# The-Model

<!-- Header dengan gradient dan badge -->
<p align="center">
  <img src="https://img.shields.io/badge/OmniMeshV2-v2.0-0A0E27?style=for-the-badge&logo=ai&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">🧠 OmniMeshV2</h1>
<p align="center"><b>AI Model Profesional Universal dengan Pelatihan Adaptif dan Keamanan Terintegrasi</b></p>

<p align="center">
  <i>OmniMeshV2 menggabungkan teknik mutakhir untuk menangani berbagai format data, memproses kode secara struktural, menstabilkan pelatihan adaptif, serta memastikan keamanan output.</i>
</p>

---

## ✨ Fitur Utama

| Komponen | Deskripsi |
|----------|-----------|
| **UDIE** (Universal Data Ingestion Engine) | Membaca berbagai format file: TXT, PDF, DOC, HTML, kode (Python, JS, Java, dll) |
| **CAST** (Code-Aware Structural Tokenizer) | Tokenisasi kode dengan AST parsing menggunakan Tree-sitter |
| **ATSG** (Adaptive Training Stability Governor) | Monitor GPU/CPU real-time, menyesuaikan batch size & top-k secara adaptif |
| **Backbone** | 48 blok interleaved (Dense → Sparse MoE → eRCD) dengan Mixture of Experts |
| **CSR v2** (Constitutional Safety Router v2) | 12 prinsip umum + 6 prinsip kode, dilengkapi self-correction |
| **Mode Pelatihan** | ML klasik (TF-IDF + LogReg), Expert (fine-tuning DistilBERT), Scratch (transformer dari nol) |
| **GUI** | Antarmuka Tkinter untuk training dan inferensi |
| **Watch Mode** | Pelatihan ulang otomatis saat data berubah |

---

## 📦 Instalasi

### 1. Install Dependencies Wajib

```bash
pip install torch pandas scikit-learn psutil gputil transformers datasets
```
### Opsional untuk PDF dan DOC
```bash
pip install PyPDF2 python-docx
```

### Opsional untuk Parsing Kode (Tree-sitter)
```bash
pip install tree-sitter tree-sitter-python tree-sitter-javascript
```

### GUI (Antarmuka Grafis)
```bash
python omnimesh_v2.py gui
```

### Training via CLI
```bash
# ML Classic (TF-IDF + Logistic Regression)
python omnimesh_v2.py train --mode ml_classic --data_dir ./data
```
```bash
# Expert mode (fine-tune DistilBERT)
python omnimesh_v2.py train --mode expert --data_dir ./data
```
```bash
# Scratch mode (transformer from zero)
python omnimesh_v2.py train --mode scratch --data_dir ./data
```

### Inferensi
```bash
# Dengan file konteks (PDF, kode, dll)
python omnimesh_v2.py infer --file laporan.pdf --prompt "Analisis laporan ini"
```
```bash
# Dengan prompt saja
python omnimesh_v2.py infer --prompt "Buat fungsi fibonacci di Python"
```

### Watch Mode (Auto-retrain saat data berubah)
```bash
python omnimesh_v2.py watch --data_dir ./data
```

<hr style="border: none; height: 2px; background: linear-gradient(90deg, transparent, #8A2BE2, #00BFFF, #8A2BE2, transparent); margin: 30px 0 20px 0;" />

<!-- ASCII Art Neuron (stylized) -->
<p align="center">
<code>
  ╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮<br/>
  ┃   🧠    ⚡    🔗    🧬    ⚙️    🔒    🛡️    🌐    📡   ┃<br/>
  ┃   ○─────○─────○─────○─────○─────○─────○─────○─────○   ┃<br/>
  ┃   "Sparse-Dense Backbone • 48 Layers • MoE • eRCD"   ┃<br/>
  ╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯
</code>
</p>

<!-- Status Badges Modern -->
<p align="center">
  <img src="https://img.shields.io/badge/🧠_Neural_State-Active-00FF00?style=for-the-badge&logo=ai&logoColor=white" />
  <img src="https://img.shields.io/badge/⚡_Adaptive-ATSG_Enabled-FF69B4?style=for-the-badge" />
  <img src="https://img.shields.io/badge/🔒_Safety-CSR_v2-32CD32?style=for-the-badge" />
  <img src="https://img.shields.io/badge/📦_Backbone-48_Blocks-FFD700?style=for-the-badge" />
</p>

<!-- Tagline with emoji -->
<p align="center">
  <b>🧠 Neural Mesh</b> &nbsp;|&nbsp; <b>⚡ Adaptive Inference</b> &nbsp;|&nbsp; <b>🔒 Constitutional AI</b> &nbsp;|&nbsp; <b>🔄 Universal Ingestion</b>
</p>

<p align="center">
  <i>“Menggabungkan kekuatan sparse-dense backbone dengan keamanan terintegrasi”</i>
</p>

<!-- Action Buttons (masih bisa diklik) -->
<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/📖_Documentation-Online-0077B5?style=for-the-badge&logo=readthedocs&logoColor=white" /></a>
  <a href="#"><img src="https://img.shields.io/badge/🐞_Report_Issue-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" /></a>
  <a href="#"><img src="https://img.shields.io/badge/💬_Community-Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" /></a>
</p>

<!-- Visitor counter dengan gaya neural -->
<p align="center">
  <img src="https://komarev.com/ghpvc/?username=omnimeshv2&label=Neural%20Activations&color=8A2BE2&style=flat-square&labelColor=0A0E27" alt="Neural Activations" />
</p>

<!-- Copyright -->
<p align="center">
  <b>Made with 🧠 by OmniMeshV2 Team | MIT License</b>
</p>
