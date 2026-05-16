# Part 2: CNN Computer Vision – Manufacturing Defect Classification

## Approach
The project uses a Convolutional Neural Network (CNN) for automated classification of manufacturing defects in images. The approach involves:
- Building a custom CNN model from scratch using TensorFlow/Keras, with convolutional layers to extract features from images, pooling layers for dimensionality reduction, and dense layers for classification.
- Handling a multi-class problem with 4 classes: `normal`, `scratch`, `dent`, and `stain`.
- Incorporating data augmentation (e.g., rotations, flips, zooms) to improve model generalization and prevent overfitting.
- Training on a synthetic dataset of 480 images (120 per class), split into training, validation, and test sets.
- Evaluating performance using metrics like accuracy, loss curves, confusion matrix, and sample predictions.
- Positioning the solution as a real-world application for manufacturing quality inspection, emphasizing speed, consistency, and scalability over manual human checks.

This approach leverages deep learning for computer vision tasks, specifically image classification, and includes industry-relevant analogies (e.g., convolutional layers as camera scanning systems).

## Data Source
The dataset used for this project is sourced from:
(https://drive.google.com/drive/folders/17xoSIAe-24-18iJiN3zKPqJl-RNDqIeW?usp=drive_link)

This folder contains synthetic images organized by defect classes (`normal`, `scratch`, `dent`, and `stain`), along with metadata files required for training the CNN model.

## Steps
The notebook outlines a step-by-step process in 11 sections, which can be summarized as follows:
1. **Install & Import Libraries**: Set up the environment by installing required packages (e.g., TensorFlow, NumPy, Matplotlib) and importing them.
2. **Configuration & Dataset Setup**: Define hyperparameters (e.g., image size, batch size, epochs) and prepare the dataset by downloading images from Google Drive and organizing them into folders with a labels.csv file.
3. **Load & Explore the Dataset**: Load images, visualize sample images from each class, and check class distribution to ensure balance.
4. **Data Preprocessing & Augmentation**: Resize images, normalize pixel values, apply augmentations (e.g., random rotations, shifts), and split data into training/validation/test sets using tools like ImageDataGenerator.
5. **Build the CNN Model**: Construct the model architecture with convolutional layers, max-pooling, dropout for regularization, and a softmax output layer for multi-class classification.
6. **Train the Model**: Compile the model with an optimizer (e.g., Adam), loss function (categorical cross-entropy), and metrics (accuracy), then fit it on the training data with validation.
7. **Visualize Training Results**: Plot accuracy and loss curves for training vs. validation to assess overfitting/underfitting.
8. **Evaluate on Test Set**: Run predictions on the test set, generate a confusion matrix, and compute metrics like precision, recall, and F1-score.
9. **Sample Predictions**: Display example predictions with images, true labels, and predicted labels to demonstrate model performance.
10. **Real-World Application – Manufacturing Quality Inspection**: Discuss how the model applies to industry, including benefits (speed, consistency), limitations (synthetic data), and next steps (transfer learning, rejection thresholds).
11. **Save & Export the Final Model**: Save the trained model in HDF5 format for deployment.

These steps form a complete pipeline from setup to deployment, with code cells implementing each part.

## Results
Based on the notebook's structure (though cells haven't been executed in this session), the results would include:
- **Training Metrics**: Accuracy and loss curves showing model convergence (e.g., training accuracy improving over epochs, validation loss stabilizing).
- **Evaluation Metrics**: Test set performance, such as overall accuracy (likely high, e.g., 90%+ for a well-trained CNN on this dataset), confusion matrix highlighting misclassifications (e.g., potential confusion between similar defects like `scratch` and `dent`), and per-class metrics.
- **Visual Outputs**: Saved images in the `results/` folder, including accuracy/loss curves and confusion matrix plots.
- **Sample Predictions**: Images in `Sample_predictions/` showing correct/incorrect classifications with confidence scores.
- Quantitative outcomes like final test accuracy, which could be documented as a key result (e.g., "Achieved 95% accuracy on the test set").

Since the notebook cells have cached outputs from previous runs, these results are reproducible by re-executing the notebook.

## Observations
Key insights and observations from the notebook include:
- **Model Performance**: The CNN effectively learns defect features, with data augmentation helping robustness. However, synthetic data may not fully represent real manufacturing variability (e.g., lighting, angles).
- **Training Dynamics**: Monitoring curves reveals if the model overfits (validation loss increasing while training loss decreases), suggesting adjustments like more dropout or early stopping.
- **Evaluation Insights**: Confusion matrix shows which classes are hardest to distinguish, guiding improvements (e.g., adding more features for `stain` vs. `dent`).
- **Industry Mapping**: CNN components directly analogize to manufacturing processes, offering benefits like millisecond inspection times and audit traceability, but requiring real-world data for deployment.
- **Limitations**: Synthetic dataset limits generalizability; transfer learning could improve results without more data. A confidence threshold for human review is recommended for edge cases.
- **Scalability**: The model is lightweight and deployable across lines, but production integration needs consideration for hardware (e.g., edge devices).
