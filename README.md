# Weather Images Classification

Weather Images Classification is a machine learning project that aims to classify images based on different weather conditions using Convolutional Neural Networks (CNNs). This repository contains the code and resources needed to train and test a model for this task.

## Features

- Preprocessing weather image datasets.
- Building and training CNN models for classification.
- Evaluating model performance with accuracy and loss metrics.
- Easily customizable for new datasets or additional weather categories.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MarawanHassaan/weather-images-classification.git
   cd weather-images-classification
   ```

2. Set up a virtual environment (optional but recommended):

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

1. Download the dataset of weather images from your preferred source. If you don’t have a dataset, you can search for one on platforms like [Kaggle](https://www.kaggle.com/) or [Google Datasets](https://datasetsearch.research.google.com/).

2. Organize the dataset into the following directory structure:

   ```
   dataset/
   ├── train/
   │   ├── sunny/
   │   ├── rainy/
   │   ├── cloudy/
   │   └── snowy/
   ├── test/
   │   ├── sunny/
   │   ├── rainy/
   │   ├── cloudy/
   │   └── snowy/
   ```

3. Update the dataset path in the configuration or code if necessary.

## Usage

### Training the Model

1. Run the training script:

   ```bash
   python train.py
   ```

2. Customize the hyperparameters (e.g., learning rate, batch size, epochs) by modifying the `config.py` file.

### Testing the Model

1. Once the model is trained, evaluate its performance on the test dataset:

   ```bash
   python test.py
   ```

2. The script will output the classification accuracy and generate a confusion matrix.

### Predicting New Images

1. Use the `predict.py` script to classify a single image:

   ```bash
   python predict.py --image path/to/image.jpg
   ```

2. The script will output the predicted weather condition.

## Directory Structure

```
weather-images-classification/
├── dataset/             # Directory for datasets
├── models/              # Saved models
├── notebooks/           # Jupyter notebooks for experiments
├── scripts/             # Python scripts for training, testing, and prediction
├── utils/               # Utility functions
├── config.py            # Configuration file for hyperparameters
├── train.py             # Script for training the model
├── test.py              # Script for testing the model
├── predict.py           # Script for predicting a single image
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
```

## Dependencies

Make sure to install the dependencies listed in `requirements.txt`. The key libraries include:

- Python 3.7+
- TensorFlow or PyTorch
- NumPy
- Matplotlib
- scikit-learn
- OpenCV (optional, for image preprocessing)

## Contributing

Contributions are welcome! If you want to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a pull request.
