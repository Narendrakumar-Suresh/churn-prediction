# Churn Prediction using PyTorch

This project is a churn prediction model built using PyTorch. It trains a neural network to predict customer churn based on input features.

## Dataset
The dataset is expected to be a CSV file named `final_dataset.csv`, where:
- Features (independent variables) are all columns except `Churn`.
- The target variable (`Churn`) is a binary column indicating whether a customer churned (1) or not (0).

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install torch pandas
```

## Model Architecture
The neural network consists of:
- Input layer with 5 features.
- Three hidden layers with ReLU activation functions.
- Output layer with a sigmoid activation function for binary classification.
- Uses `BCELoss` as the loss function and `SGD` optimizer.

## Training
The dataset is split into 80% training and 20% testing. Training is performed over 100 epochs with a batch size of 50. The loss is printed every 100 steps.

## Usage
1. Place `final_dataset.csv` in the project directory.
2. Run the script:
   ```bash
   python model.py
   ```
3. The final accuracy will be displayed at the end of execution.

## Evaluation
After training, the model is evaluated on the test set, and accuracy is calculated based on correct predictions.

## Possible Improvements
- Experiment with different optimizers (Adam, RMSprop, etc.).
- Tune hyperparameters like learning rate, batch size, and hidden layer size.
- Add dropout layers to prevent overfitting.
- Implement feature scaling for better convergence.
