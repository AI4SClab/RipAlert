# RipAlert: A Future-Frame-Aware Framework for Rip Current Forecasting and Early Alerting

[![Paper](https://img.shields.io/badge/Paper-AAAI_2026-red)](./AAAI_Press_Formatting_Instructions_for_Authors_Using_LaTeX.pdf)
[![Code](https://img.shields.io/badge/Code-Repo-blue)](https://aaai.org/example/code)
[![Dataset](https://img.shields.io/badge/Dataset-Download-green)](https://aaai.org/example/datasets)

[cite_start]This is the official repository for the paper: **"RipAlert: A Future-Frame-Aware Framework for Rip Current Forecasting and Early Alerting"**[cite: 1].

[cite_start]RipAlert is a novel framework designed to forecast near-future coastal dynamics and proactively identify rip current risks[cite: 14]. [cite_start]Unlike traditional reactive methods that detect rip currents only *after* they form, RipAlert leverages temporal motion patterns to detect them **up to 5 seconds before they become visibly apparent**[cite: 15].

[cite_start]The system has been successfully deployed at high-risk beaches in China, issuing effective early warnings for real-world events[cite: 18].

## 🌟 Core Features

* [cite_start]**Proactive Forecasting:** Predicts future frames to enable detection up to 5 seconds before a rip current visibly forms[cite: 15].
* [cite_start]**Region-Sensitive Optical Flow:** A lightweight video prediction module based on optical flow techniques forecasts the evolution of coastal scenes[cite: 51]. [cite_start]It is specifically designed to capture subtle, early-stage reverse-flow anomalies[cite: 15].
* [cite_start]**Entropy-Based Detector:** We introduce a **Content-Aware Entropy Attention (CEA)** module into a YOLOv12-based detector[cite: 53, 162]. [cite_start]CEA dynamically allocates computational resources, using deformable attention for high-entropy (complex) regions like turbulent water and efficient depth-wise convolutions for low-entropy (simple) regions like calm sea or sand[cite: 163].
* [cite_start]**Real-World Deployment:** The framework is implemented as a lightweight mobile application to support timely alerts in resource-limited beach settings[cite: 16, 48].
* [cite_start]**Public Dataset:** We provide a new, curated dataset of over 2,000 annotated images focused on early-stage rip current patterns[cite: 16, 54].

## 🔧 Framework Overview

[cite_start]The RipAlert framework consists of four sequential stages[cite: 71]:



1.  [cite_start]**(a) Data Collection:** Video streams are acquired from drones and coastal cameras at 2 FPS[cite: 72].
2.  [cite_start]**(b) Region-Sensitive Optical Flow Prediction:** A dual-frame optical flow algorithm estimates pixel-wise motion[cite: 73]. [cite_start]This flow field is segmented into `static`, `turbulent`, and `reverse-flow` regions based on magnitude and direction[cite: 73]. [cite_start]Motion vectors are adjusted based on these semantic regions to synthesize future frames that emphasize early-stage rip current dynamics[cite: 74].
3.  [cite_start]**(c) Entropy-Based Detector:** Both historical and the semantically-enhanced predicted frames are fed into our improved YOLOv12 detector[cite: 75]. [cite_start]The Content-Aware Entropy Attention (CEA) module dynamically adjusts spatial focus based on motion complexity[cite: 75].
4.  [cite_start]**(d) End-to-End Application:** Detection outputs are integrated into a lightweight mobile application that provides real-time alerts, supporting beach rescue and disaster prevention[cite: 76, 117].

## ⚙️ Installation

1.  Clone the repository:
    ```bash
    git clone [https://aaai.org/example/code](https://aaai.org/example/code)
    cd RipAlert
    ```
2.  Create a virtual environment and install dependencies:
    ```bash
    # We recommend using Python 3.8+
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt 
    ```
    [cite_start]*(Note: All models are implemented in PyTorch [cite: 205])*

## 📊 Datasets

[cite_start]Our framework is trained and evaluated on two datasets[cite: 200]:

1.  [cite_start]**Rip Current Segmentation Benchmark (RipVIS):** Provides 2,466 annotated rip-current images, 1,307 negatives, and 17 test videos[cite: 201].
2.  [cite_start]**Our Custom Dataset:** We built a new dataset of 2,143 annotated frames collected from public sources and our own drone/shore surveillance[cite: 202]. [cite_start]This dataset is specifically focused on **early-stage rip current patterns** and will be publicly released[cite: 203, 204].

[cite_start]You can download our curated dataset here: **[https://aaai.org/example/datasets](https://aaai.org/example/datasets)** [cite: 21]

## 🚀 Usage

### Training

You can train the modules separately.

1.  **Train the Region-Sensitive Optical Flow Module:**
    [cite_start]*(Trained for 30 epochs with Adam optimizer, LR 0.001 [cite: 211])*
    ```bash
    python train_flow.py --config configs/flow_config.yaml
    ```

2.  **Train the Entropy-Based Detector:**
    [cite_start]*(Trained for 50 epochs with batch size 16, one-cycle LR schedule starting at 1e-4 [cite: 216])*
    ```bash
    python train_detector.py --config configs/detector_config.yaml
    ```

### Evaluation

To evaluate a trained model, run:

```bash
python evaluate.py --model_type detector \
                   --weights /path/to/your/detector_weights.pt \
                   --data data/ripalert_custom.yaml
````

## 📈 Results

### 1\. Prediction Quality

[cite\_start]Our region-sensitive optical flow method produces higher-quality future frames with better structural consistency compared to standard methods[cite: 251, 264].

[cite\_start]**Table 1: Quantitative evaluation of predicted frame quality** [cite: 253]

| Method | PSNR (↑) | SSIM (↑) |
| :--- | :---: | :---: |
| TV-L1 | 30.6631 | 0.7226 |
| Lucas-Kanade | 30.6226 | 0.7168 |
| **Region-Sensitive (Ours)** | **30.8708** | **0.7539** |

### 2\. Detection Performance

[cite\_start]Our full model, integrating the CEA module, achieves state-of-the-art performance, significantly outperforming all baselines[cite: 275]. [cite\_start]The CEA module alone boosts the mAP50 from 90.96% (baseline YOLOv12) to **94.68%**[cite: 290].

[cite\_start]**Table 2: Performance comparison on the detection datasets** [cite: 274]

| Model | Precision (↑) | Recall (↑) | mAP50 (↑) | mAP50-95 (↑) |
| :--- | :---: | :---: | :---: | :---: |
| YOLOv8s-seg | 0.8762 | 0.8467 | 0.8925 | 0.4666 |
| YOLOv8m-seg | 0.8804 | 0.8504 | 0.8992 | 0.4748 |
| RT-DETRv2 | 0.8424 | 0.7590 | 0.8770 | 0.4290 |
| YOLOv12 (Baseline) | 0.9067 | 0.8521 | 0.9096 | 0.4788 |
| **Ours (RipAlert + CEA)** | **0.9336** | **0.8844** | **0.9468** | **0.4849** |

## 📱 Deployment and Social Impact

[cite\_start]The RipAlert system has been piloted in collaboration with the Institute of Oceanology, Chinese Academy of Sciences (IOCAS) at high-risk coastal zones in Shandong and Fujian, China[cite: 309, 311].

  * [cite\_start]It enables **preemptive detection 5-10 seconds in advance**[cite: 311].
  * [cite\_start]A companion mobile app provides real-time alerts to beachgoers and supervisors[cite: 312, 317].
  * [cite\_start]The system has successfully issued alerts for 32 real-world rip current events[cite: 316].

## 📜 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{wan2026ripalert,
  title={RipAlert: A Future-Frame-Aware Framework for Rip Current Forecasting and Early Alerting},
  author={Wan, Meng and Su, Qi and Xia, Zhixin and Chen, Kanglin and Wang, Jue and Liu, Tiantian and Cao, Rongqiang and Cui, Hui and Shi, Peng and Wang, Yangang and Feng, Liqiang and Zhao, Zhenbing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

