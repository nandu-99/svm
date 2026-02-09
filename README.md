# Floor Segmentation using Classical Machine Learning

This project explores **classical computer vision and machine learning approaches** for floor segmentation using a COCO-style annotated dataset.
The goal is to **systematically study how feature representations affect segmentation performance**, progressing from simple pixel-level features to more structured representations.

The work is organized into **four main approaches**, each implemented in a separate notebook for clarity and fair comparison.

---

## Dataset

This project uses the **Floor Segmentation dataset** hosted on Roboflow.

Dataset link:
[https://universe.roboflow.com/uni-phtjr/floor-segmentation-huuot](https://universe.roboflow.com/uni-phtjr/floor-segmentation-huuot)

* **Format:** COCO segmentation format
* **Task:** Binary segmentation

  * `1` → Floor
  * `0` → Non-floor
* **Annotations:** Pixel-level masks for floor regions
* **Images:** Indoor scenes with varying lighting, textures, and layouts

The dataset can be downloaded from the above link and used directly in Google Colab or a local environment.

---

## Approach Overview

### Approach 1 — Pixel-level RGB Classification

Each pixel is treated as an independent sample and classified using only its RGB color values.

* **Feature Vector:** (R, G, B)
* **Sample Unit:** Pixel
* **Model:** SVM

This approach serves as a simple baseline and does not use any spatial or contextual information.

**Observations:**

* Highly sensitive to lighting variations
* No spatial or texture awareness
* Produces noisy segmentation masks

<img width="857" height="273" alt="Screenshot 2026-02-09 at 9 34 53 PM" src="https://github.com/user-attachments/assets/d7b9ffda-7b26-49d2-ba2a-dd1950db8a8d" />

---

### Approach 2 — Pixel-level RGB + Spatial Coordinates

This approach augments pixel-level RGB features with normalized spatial coordinates.

* **Feature Vector:** (R, G, B, x, y)
* **Sample Unit:** Pixel
* **Model:** SVM

Adding spatial priors allows the model to learn location-dependent patterns, such as the tendency of floors to appear near the bottom of images.

**Observations:**

* Significant improvement over RGB-only classification
* Cleaner segmentation compared to Approach 1
* Still ignores texture and local structure

<img width="857" height="271" alt="Screenshot 2026-02-09 at 9 35 07 PM" src="https://github.com/user-attachments/assets/8a0c0d85-165d-4fe5-b408-38524c1beab9" />

---

### Approach 3 — Region-based Mean RGB + Spatial Coordinates

Instead of classifying individual pixels, the image is divided into fixed-size regions, and each region is treated as a single sample.

* **Feature Vector:** (mean R, mean G, mean B, mean x, mean y)
* **Sample Unit:** Region
* **Model:** SVM

Region-level aggregation reduces pixel noise and stabilizes predictions.

**Observations:**

* Best balance between simplicity and performance
* Robust to local pixel noise
* Produces block-like segmentation boundaries

<img width="855" height="271" alt="Screenshot 2026-02-09 at 9 35 37 PM" src="https://github.com/user-attachments/assets/22d36da5-92f9-425f-b906-1c351b2b82d8" />


---

### Approach 4 — HOG-based Patch Classification

This approach represents image patches using **Histogram of Oriented Gradients (HOG)**, focusing on texture and shape rather than color.

* **Feature Vector:** HOG descriptor
* **Sample Unit:** Patch
* **Model:** SVM

Two variants were evaluated:

**Small dataset variant**

* High training accuracy
* Poor test performance due to overfitting

**Large dataset variant**

* Improved generalization with sufficient training data
* Strong recall for floor regions

**Observations:**

* HOG features are highly discriminative but data-hungry
* Ignores color information
* Sensitive to class imbalance


<img width="857" height="270" alt="Screenshot 2026-02-09 at 9 35 50 PM" src="https://github.com/user-attachments/assets/8f8dc0c2-1ca1-45e2-832f-320bef8e817b" />

<img width="926" height="209" alt="Screenshot 2026-02-09 at 4 05 28 PM" src="https://github.com/user-attachments/assets/b5183cc5-71b8-4a16-a0de-148425b14f09" />


---

## Quantitative Results Summary

| Approach                | Features                | Test Accuracy | Generalization |
| ----------------------- | ----------------------- | ------------- | -------------- |
| Approach 1              | RGB                     | ~0.66         | Poor           |
| Approach 2              | RGB + (x,y)             | ~0.86         | Good           |
| Approach 3              | Region mean RGB + (x,y) | ~0.91         | Very Good      |
| Approach 4 (small data) | HOG                     | ~0.70         | Overfitting    |
| Approach 4 (large data) | HOG                     | ~0.95         | Strong         |

---

## Role of SVM Hyperparameter C and Kernel Choice

### Role of the Regularization Parameter C

The parameter **C** in Support Vector Machines controls the trade-off between **margin maximization** and **classification error on training data**.

* **Low C**

  * Encourages a wider margin
  * Allows more misclassifications
  * Leads to simpler decision boundaries
  * Reduces overfitting but may underfit

* **High C**

  * Penalizes misclassification strongly
  * Forces the model to fit training data closely
  * Produces complex decision boundaries
  * Increases risk of overfitting

**Observed behavior in this project:**

* In pixel-level approaches (Approach 1 and 2), moderate values of C worked best due to noisy data.
* In region-based features (Approach 3), the model was less sensitive to C because features were more stable.
* In HOG-based classification (Approach 4), high C values led to severe overfitting, especially with limited training data.

---

### Comparison of Kernel Functions

Different SVM kernels were evaluated to understand how feature representations interact with decision boundaries.

#### Linear Kernel

* Learns a linear decision boundary
* Computationally efficient
* Works well when features are already discriminative

**Best suited for:**

* Pixel RGB + (x, y) features
* Region-based mean RGB features

**Observed behavior:**

* Stable training
* Minimal overfitting
* Good generalization in Approaches 2 and 3

---

#### RBF (Radial Basis Function) Kernel

* Learns non-linear decision boundaries
* Highly expressive
* Sensitive to hyperparameters and dataset size

**Best suited for:**

* Texture-based features such as HOG
* Non-linearly separable data

**Observed behavior:**

* Overfitting on small HOG datasets
* Strong performance when trained with sufficient samples
* High training accuracy with a noticeable train–test gap when data is limited

---

#### Polynomial Kernel

* Models polynomial relationships between features
* More complex than linear but less flexible than RBF

**Observed behavior:**

* Did not provide consistent improvements
* Increased training time
* Tended to overfit without clear gains

---

### Summary of Kernel Behavior in This Project

| Kernel     | Complexity  | Stability | Best Use Case                        |
| ---------- | ----------- | --------- | ------------------------------------ |
| Linear     | Low         | High      | RGB + spatial, region-based features |
| RBF        | High        | Medium    | HOG features with sufficient data    |
| Polynomial | Medium–High | Low       | Not recommended for this task        |

---

## Key Insights

* Raw RGB features alone are insufficient for reliable floor segmentation.
* Spatial information significantly improves performance.
* Region-based aggregation provides the best trade-off between robustness and simplicity.
* HOG features perform well when sufficient training data is available.
* Classical methods benefit greatly from structured feature design.

---

## Project Structure

```
├── approach1_pixel_rgb.ipynb
├── approach2_pixel_rgb_xy.ipynb
├── approach3_region_rgb_xy.ipynb
├── approach4_hog_small_dataset.ipynb
├── approach4_hog_large_dataset.ipynb
├── README.md
```

---

## Conclusion

This project demonstrates a systematic progression from simple to more structured feature representations for floor segmentation. The experiments show that adding spatial context and region-level abstraction leads to significant performance gains. While HOG features are powerful, their effectiveness depends on dataset size and label quality.

---

## Future Work

* Combine HOG and color features for hybrid representations
* Address class imbalance using sampling strategies
* Compare against deep learning-based segmentation models
* Evaluate cross-dataset generalization

---
