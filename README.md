# 🧠 AI-Assisted Detection of Tuberculosis, Pneumonia & Normal Cases

This repository presents my research project on developing an **AI-assisted diagnostic model** for detecting **Tuberculosis**, **Pneumonia**, and **Normal** chest X-rays using **Deep Learning**.  
The model utilizes a **fine-tuned ResNet50** architecture enhanced with an **Adaptive Categorical Focal Loss** to tackle class imbalance and improve generalization.

---

## 🎯 Project Overview

Chest radiography remains a primary diagnostic tool for respiratory diseases such as Tuberculosis (TB) and Pneumonia.  
However, manual interpretation is **time-consuming** and **subject to variability** among clinicians, especially in **low-resource healthcare settings**.

To address this, I developed a **computer-aided diagnostic (CAD)** system using deep learning that can automatically classify chest X-rays into:
- **Normal**
- **Pneumonia**
- **Tuberculosis**

This work lies within the intersection of **AI and medical imaging**, aligning with the **Medical Image Analysis (MIA)** research focus.

---

## 🧩 Datasets

Three open-access datasets were combined to achieve robust class diversity:

| Dataset Source | Description | Usage |
|----------------|--------------|-------|
| [Chest X-Ray Pneumonia Dataset (Kaggle)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) | Normal & Pneumonia X-rays | Training/Validation |
| [Tuberculosis Chest Radiography Dataset (Kaggle)](https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-radiography-database) | TB & Normal X-rays | Training/Validation |
| [COVID-19 Radiography Dataset (Kaggle)](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) | Viral Pneumonia & Normal X-rays | Data Augmentation |

**Data Split:**
- 70% Training  
- 15% Validation  
- 15% Testing  

**Classes:**
- Normal  
- Pneumonia  
- Tuberculosis  

---

## 🧠 Model Architecture

**Base Model:** ResNet50 (pretrained on ImageNet)

The network was customized for medical image classification:


Initially, the base model was frozen to leverage ImageNet features.  
Later, the final **20 layers** were unfrozen for **fine-tuning**, enabling the network to learn domain-specific radiological patterns.

---

## ⚙️ Adaptive Categorical Focal Loss

To mitigate **class imbalance** (particularly underrepresented TB cases), I implemented a **custom adaptive focal loss** that dynamically adjusts the parameters \( \alpha \) and \( \gamma \) based on per-class recall during training.

**Loss Function:**

\[
FL(p_t) = - \alpha_t (1 - p_t)^{\gamma_t} \log(p_t)
\]

Where:
- \( p_t \): Predicted probability of the true class  
- \( \alpha_t \): Adaptive class weight (updates per epoch)  
- \( \gamma_t \): Focusing parameter emphasizing hard-to-classify examples  

This adaptive mechanism improves recall for minority classes without sacrificing precision.

---

## 🚀 Training Strategy

- **Optimizer:** Adam (LR = 0.001 → adaptive reduction)
- **Batch Size:** 64  
- **EarlyStopping:** Patience = 4 (restore best weights)  
- **ModelCheckpoint:** Saves best model on validation loss  
- **Fine-Tuning Phase:** Unfreeze last 20 ResNet50 layers  

---

## 📊 Results

| Class        | Precision | Recall | F1-Score | Support |
|---------------|------------|---------|-----------|----------|
| Normal        | 1.00 | 1.00 | 1.00 | 2255 |
| Pneumonia     | 1.00 | 0.99 | 0.99 | 783 |
| Tuberculosis  | 0.99 | 1.00 | 0.99 | 105 |
| **Overall Accuracy** | **1.00** | **1.00** | **1.00** | **3143** |

✅ **Macro Avg F1:** 0.996  
✅ **Weighted Avg F1:** 0.999  
✅ **Test Accuracy:** ~99.9%

The model demonstrates near-perfect performance while maintaining high generalization across datasets.

---

## 🔬 Interpretability

Although Grad-CAM was not applied in this version, the model’s structure supports post-hoc explainability using:
- **Grad-CAM / Grad-CAM++**
- **Integrated Gradients**

These can be incorporated to visualize class activation regions for clinical interpretability.

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|----------|
| TensorFlow / Keras | Deep Learning Framework |
| NumPy, Pandas | Data Handling |
| Matplotlib, Seaborn | Visualization |
| scikit-learn | Evaluation Metrics |
| Kaggle | Dataset Hosting & Experimentation |

---

## 📁 Repository Structure


---

## 🎓 Research Significance

This project highlights how **transfer learning + adaptive loss functions** can address **class imbalance** in medical imaging tasks.  
The approach provides a **scalable diagnostic tool** for healthcare systems facing radiologist shortages, aligning with global health initiatives in **AI for Medicine**.

---

## 👨‍💻 Author

**Hamza Shahid**  
Bachelor of Biomedical Engineering (with Distinction)  
University of Engineering & Technology (UET), Lahore  

🔍 Research Interests:  
AI in Healthcare • Medical Image Analysis • Deep Learning for Diagnostics  

---

## 📜 License

This repository is released under the **MIT License** — feel free to use and build upon it for academic and research purposes.

---

## 🌍 Acknowledgment

Special thanks to **Kaggle community datasets** and open-access AI researchers whose contributions enable transparent, reproducible science.


