## Hybrid Anomaly Detection using Autoencoder + NPLM

### *Introduction*

This repository provides an advanced hybrid anomaly detection model that combines an *Autoencoder* with the *New Physics Learning Machine (NPLM).* This approach is designed to detect both obvious and subtle anomalies in **time-series data*, offering a robust solution for real-world anomaly detection challenges.

The concept is inspired by the research paper *"Robust Resonant Anomaly Detection with NPLM"* by *Gaia Grosso, Debajyoti Sengupta, Tobias Golling, and Philip Harris.* By leveraging the strengths of *deep learning* (Autoencoder) and *physics-inspired models* (NPLM), this hybrid model aims to enhance anomaly detection performance across diverse applications.

- *Primary Developer*: Giri D  
- *GitHub Repository*: [Hybrid Anomaly Detection using Autoencoder + NPLM](https://github.com/vbxll/-Hybrid-Anomaly-Detection-using-Autoencoder-NPLM)

---

### *Why This Dataset?*

The dataset used for this project comes from the Kaggle competition *[Demand Forecasting - Kernels Only](https://www.kaggle.com/competitions/demand-forecasting-kernels-only)*. This dataset contains daily sales data for multiple stores and items over a given period.

Key reasons for selecting this dataset include:

1. *Realistic Scenario*: Demand patterns across various stores and products offer a real-world scenario for detecting anomalies.
2. *Time-Series Nature*: Its sequential nature makes it ideal for applying advanced anomaly detection techniques.
3. *Business Insights*: Anomalies in sales data can signal significant events such as stockouts, promotions, or errors.

This hybrid model effectively captures both noticeable and subtle anomalies in sales patterns, making it highly relevant for demand forecasting.

---

### *Methodology*

#### *Hybrid Approach*

The hybrid approach integrates:
- *Autoencoder*: For filtering obvious anomalies.  
- *NPLM*: For detecting more subtle, complex anomalies.

This two-step process ensures a comprehensive and robust detection of anomalies in the dataset.

---

### *Step 1: Autoencoder for Obvious Anomalies*

#### *Model Architecture*

The Autoencoder is designed to encode input data into a lower-dimensional representation and reconstruct it. The reconstruction error is computed using *Mean Squared Error (MSE)*, and anomalies are identified by applying a threshold (95th percentile) on the error.

*Autoencoder Model Summary*:

| Layer (Type)   | Output Shape    | Param #  |
|----------------|-----------------|----------|
| *Linear-1*   | [-1, 1, 64]     | 3,264    |
| *ReLU-2*     | [-1, 1, 64]     | 0        |
| *Linear-3*   | [-1, 1, 32]     | 2,080    |
| *ReLU-4*     | [-1, 1, 32]     | 0        |
| *Linear-5*   | [-1, 1, 64]     | 2,112    |
| *ReLU-6*     | [-1, 1, 64]     | 0        |
| *Linear-7*   | [-1, 1, 50]     | 3,250    |
| *Sigmoid-8*  | [-1, 1, 50]     | 0        |
| *Total Params* | *10,706* |   |

---

### *Step 2: NPLM for Subtle Anomalies*

The *NPLM* is used to detect more subtle anomalies that the Autoencoder might miss. The filtered data from the Autoencoder is used as input to train the NPLM, which outputs an *anomaly score* for each sample. A threshold (95th percentile) is applied on the scores to identify subtle anomalies.

#### *NPLM Model Summary*:

| Layer (Type)   | Output Shape    | Param #  |
|----------------|-----------------|----------|
| *Linear-1*   | [-1, 1, 64]     | 3,264    |
| *ReLU-2*     | [-1, 1, 64]     | 0        |
| *Linear-3*   | [-1, 1, 32]     | 2,080    |
| *ReLU-4*     | [-1, 1, 32]     | 0        |
| *Linear-5*   | [-1, 1, 1]      | 33       |
| *Total Params* | *5,377* |   |

---

### *Results*

1. *Filtered Data*: The Autoencoder effectively filters out obvious anomalies, leaving a refined dataset for NPLM processing.
2. *Detected Anomalies*: The NPLM successfully identifies additional subtle anomalies that were not captured by the Autoencoder.

*Visualization*:  
- Interactive visualizations such as histograms of anomaly scores and scatter plots of detected anomalies are generated using *Plotly*.

---

### *Applications*

This hybrid anomaly detection model has a wide range of potential applications, including:

1. *Retail*: Detecting unusual sales patterns, stockouts, or promotional effects.  
2. *Finance*: Identifying irregular trading patterns or fraudulent transactions.  
3. *Manufacturing*: Monitoring equipment data for predictive maintenance.  
4. *Healthcare*: Detecting anomalies in patient health metrics.

---

### *Citation*

#### *NPLM Research*

If you use the NPLM concept from this project, please cite the original paper:

- Gaia Grosso, Debajyoti Sengupta, Tobias Golling, and Philip Harris, "Robust Resonant Anomaly Detection with NPLM," arXiv preprint [arXiv:2501.01778](https://arxiv.org/abs/2501.01778).

#### *This Repository*

If you use this hybrid approach, please cite the following:

- Giri D, "Hybrid Anomaly Detection using Autoencoder + NPLM," GitHub Repository [Link](https://github.com/vbxll/-Hybrid-Anomaly-Detection-using-Autoencoder-NPLM).  
  *Designation*: CTO of Vybron, Head of Quantum Computing at AIQUBIT, Lead AI Developer at KAHE's R&D Sector.

---

### *How to Run*

Clone the repository:

bash

git clone https://github.com/vbxll/-Hybrid-Anomaly-Detection-using-Autoencoder-NPLM.git
Install the required libraries:

bash

pip install torch torchvision pandas matplotlib numpy plotly nbformat

3. *Update file paths* for the training and testing datasets.

4. *Run the script* to train the model and visualize results.

---

### *License*

This project is licensed under the *MIT License*. You are free to use, modify, and distribute the code. If you publish any work using this model, please provide proper citation.

---

### *Contact*

For any questions or collaboration opportunities, feel free to reach out:

- *GitHub*: [vbxll](https://github.com/vbxll)  
- *Email*: [giri03officail@gmail.com](mailto:giri03officail@gmail.com)  
- *LinkedIn*: [Giri D](https://www.linkedin.com/in/giri-d-nssp)

