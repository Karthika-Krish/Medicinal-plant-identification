
🌿 Medicinal Plant Identification using Machine Learning

Introduction

Medicinal plants play a crucial role in healthcare systems worldwide. Identifying these plants accurately is essential for their effective use in medicine. This project leverages machine learning (ML), specifically deep learning techniques, to classify medicinal plants based on images.

Objectives

To develop a machine learning model that can identify medicinal plants from images.

To utilize Convolutional Neural Networks (CNNs) for feature extraction and classification.

Tools & Technologies Used

Programming Language: Python

Deep Learning Frameworks: TensorFlow, Keras, PyTorch

Data Handling & Visualization: Pandas, NumPy, Matplotlib, Seaborn

Image Processing: OpenCV

Dataset Source: Kaggle (Medicinal Plant Image Dataset)

Model Architectures: CNN, MobileNet

Development Environment: Google Colab, Jupyter Notebook

Version Control: Git, GitHub

Dataset The dataset used in this project is sourced from Kaggle and contains images of different medicinal plant species. It is split into:

Training Set: 80% of images

Test Set: 20% of images

Data Preprocessing Steps:

Resizing images to a fixed dimension (e.g., 224x224 pixels)

Normalization of pixel values

Data augmentation (rotation, flipping, zooming) to improve model generalization

Model Development

Step 1: Data Preparation

Apply preprocessing transformations.

Step 2: Model Selection

Custom CNN Model – A simple CNN with convolutional, pooling, and fully connected layers.

Transfer Learning – Using pre-trained models like MobileNet.

Step 3: Model Training

Loss Function: Categorical Crossentropy

Evaluation Metrics: Accuracy, Precision, Recall, F1-score

Step 4: Model Evaluation

Use Confusion Matrix to visualize classification results.

Compute accuracy on the test dataset.

Implementation & Deployment

Local Testing Run the model locally using:

python src/predict.py --medicinal_plant.jpg

Current Progress & Next Steps

Collected and preprocessed a medicinal plant dataset. Implemented MobileNet Model. Built a Flask-based user interface for image upload and prediction. Integrated frontend with backend for real-time plant identification. Achieved satisfactory accuracy with smooth UI functionality.

Future Enhancements

Expanding the dataset to include more medicinal plant species.

Integrating object detection to identify multiple plants in a single image.

Conclusion

This project demonstrates the application of machine learning in plant species identification. By leveraging deep learning, we can classify medicinal plants accurately, benefiting researchers and herbal medicine practitioners.
