# Eye-patch shift Dataset

[eyeshift.ipynb](eyeshift.ipynb) contains code that identifies defects in eye-patches for shampoo packets.
First, a YOLOv8 model identifies horizontal and vertical cuts, then a linear regression is performed through the horizontal cuts,
and eye-patches outside a threshold are categorised as defective.

**Accuracy:** 92.98%

![img.png](assets/eyeshift.png)
![img.png](assets/eyeshift2.png)

# Soap Dataset

[soap-binary-classifier.ipynb](soap-binary-classifier.ipynb) uses a simple fully-connected layer on the outputs from ResNet18, 
fine-tuned on a dataset of soap to classify as either defective or non-defective.

**Accuracy:** 100%

<table>
<tr>
<td>Non-Defective</td>
<td>Defective</td>
</tr>
  <tr>
    <td> <img src="data/soap/non_defects/HT-GE232GC-T1-C-Snapshot-20240518-104031-936-621730587909.BMP"  alt="1" width = 360px height = 360px ></td>
    <td><img src="data/soap/defects/HT-GE232GC-T1-C-Snapshot-20240525-110516-295-9220658954_BMP.rf.c60b80d2f6ef4dece9cc1480d213823f.jpg" alt="2" width = 360px height = 360px></td>
   </tr>
</table>

[soap-feature-clustering.ipynb](soap-feature-clustering.ipynb) is an unsupervised approach to defect detection in 
this dataset where features from the ResNet18 output undergo a Singular Value Decomposition (SVD) and are then clustered using Birch.

**Accuracy:** 99.55%

<table>
<tr>
<td>Ground Truth</td>
<td>SVD & Clustering</td>
<td>Prediction</td>
</tr>
  <tr>
    <td> <img src="assets/soap-clustering-truth.png"  alt="1" width = 240px height = 240px ></td>
    <td><img src="assets/soap-clustering-outputs.png" alt="2" width = 240px height = 240px></td>
    <td><img src="assets/soap-clustering-prediction.png" alt="3" width = 240px height = 240px></td>
   </tr>
</table>

[soap-autoencoder.ipynb](soap-autoencoder.ipynb) is actually a U-net which attempts to reconstruct masked images of soap to predict defective pieces by correcting errors.

**Accuracy:** untested.

<table>
<tr>
<td>Masked Input</td>
<td>Prediction</td>
</tr>
  <tr>
    <td> <img src="assets/soap-ae-input.png"  alt="1" width = 360px height = 360px ></td>
    <td><img src="assets/soap-ae-prediction.png" alt="2" width = 360px height = 360px></td>
   </tr>
</table>

[soap-fourier-analysis.ipynb](soap-fourier-analysis.ipynb) is a single-shot, unsupervised method for defect detection on a normalised dataset.
A non-defective single-shot reference image is chosen and the squared complex-difference between its Fourier Transform and all other images in the dataset are compared and clustered.

**Accuracy:** 100%

<table>
<tr>
<td>Fourier Transform of Soap</td>
<td>Histogram of Differences to Reference</td>
</tr>
  <tr>
    <td> <img src="assets/soap-fourier-transform.png"  alt="1" width = 400px height = 150px ></td>
    <td><img src="assets/soap-fourier-transform-histogram.png" alt="2" width = 220px height = 150px></td>
   </tr>
</table>


# Soap Side Dataset

# Shampoo Dataset
