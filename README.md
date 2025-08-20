# 🩺 Skin Cancer Detection using CNN (TensorFlow & Keras)

This project focuses on **skin cancer detection** using a **Convolutional Neural Network (CNN)** built with **TensorFlow and Keras**.  
The dataset used is the **ISIC (International Skin Imaging Collaboration)** dataset, which contains images of different types of skin lesions.

# 🚀 Project Workflow

# 1️⃣ Importing Libraries
The project uses:
- `TensorFlow` / `Keras` (Deep Learning)
- `Matplotlib` (Visualization)
- `NumPy` & `Pandas` (Data handling)
- `Pathlib` (File handling)

# 2️⃣ Dataset
- **Train Path:** Contains labeled training images of skin lesions.
- **Test Path:** Contains test images for evaluation.
- Dataset is split into **80% training** and **20% validation**.


# 3️⃣ Data Preprocessing
- Images are resized to **180x180**.
- Batch size: **32**
- Data pipeline uses:
  - **Caching**
  - **Shuffling**
  - **Prefetching with AUTOTUNE**

# 4️⃣ Data Visualization
- Displayed sample images from different classes.
- Bar chart shows distribution of images across classes.
- Applied **data augmentation**:
  - Random flip (horizontal & vertical)
  - Random rotation
  - Random zoom

 # 5️⃣ Model Architecture (CNN)
The CNN is built using `Sequential` model:
- **Conv2D + MaxPooling** layers (32, 64, 128 filters)
- **Flatten Layer**
- **Dense Layer** (128 neurons, ReLU activation)
- **Dropout Layer** (0.5 to prevent overfitting)
- **Output Layer** (Softmax → multi-class classification)

# 6️⃣ Model Compilation & Training
- Optimizer: **Adam**
- Loss Function: **SparseCategoricalCrossentropy**
- Metrics: **Accuracy**
- Trained for **30 epochs** with validation data.

# 7️⃣ Model Evaluation
- Plotted **Training vs Validation Accuracy**
- Plotted **Training vs Validation Loss**
- Results show how the model learns over epochs.

# 8️⃣ Model Saving
The trained model is saved as:
```bash
skin_cancer_model.h5

# 📊 Results

Model achieved good accuracy on both training and validation datasets.

Data augmentation helped improve generalization.

📌 Future Improvements

Use transfer learning with EfficientNet / ResNet50 for higher accuracy.

Hyperparameter tuning (learning rate, batch size, epochs).

Deploy model as a web application using Flask/Streamlit.

🙌 Acknowledgements

Dataset: ISIC - International Skin Imaging Collaboration

Frameworks: TensorFlow & Keras
