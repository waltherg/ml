# Dimensionality Reduction using Autoencoders for Multiclass Classification with XGBoost

## Data

```bash
$ kaggle datasets download -d uciml/human-activity-recognition-with-smartphones
$ unzip human-activity-recognition-with-smartphones.zip
```

## Experiment

### Overview
In this experiment, we aimed to explore the effect of dimensionality reduction using an autoencoder on a tabular data multiclass classification task with XGBoost. Dimensionality reduction can potentially help improve the classification performance by reducing noise, speeding up the training process, and reducing overfitting. Autoencoders are unsupervised neural networks that can learn to compress data in a lower-dimensional space while retaining essential information.

### Method
We implemented an autoencoder using PyTorch and trained it to minimize the mean squared error between the input and reconstructed data. The dataset was preprocessed, and label encoding was applied to the target variable. The autoencoder was designed with a simple architecture, including two hidden layers in the encoder and two hidden layers in the decoder.

We compared the performance of XGBoost on the raw data with the performance of XGBoost on the encoded, dimensionality-reduced data obtained from the trained autoencoder.

### Results
The classification results are as follows:

Event classification without dimensionality reduction:

Dimensions before reduction: 561
- Accuracy: 0.9389
- Precision: 0.9404
- Recall: 0.9389
- F1 Score: 0.9387

Event classification with dimensionality reduction using autoencoder:

Dimensions after reduction: 73
- Accuracy: 0.8945
- Precision: 0.8958
- Recall: 0.8945
- F1 Score: 0.8945

The dimensionality reduction using the autoencoder did not improve the classification performance. In fact, the performance metrics (accuracy, precision, recall, and F1 score) were slightly lower for the dimensionality-reduced data.

### Discussion
There could be several reasons why dimensionality reduction did not improve the results:

- Information loss: The autoencoder might not have captured all the essential information from the original high-dimensional data. Some important features could be lost during the encoding process, which could impact the classification performance.

- Model architecture: The architecture of the autoencoder might not be optimal for the given dataset. The choice of the number of layers, neurons in each layer, and activation functions can significantly influence the performance of the autoencoder.

- Hyperparameters: The learning rate, number of training epochs, and optimizer might not be optimal for training the autoencoder. These hyperparameters can affect the autoencoder's convergence and the quality of the learned representations.

Further experimentation with different autoencoder architectures, hyperparameters, and regularization techniques could potentially lead to better dimensionality reduction and improved classification performance. Additionally, other dimensionality reduction techniques such as PCA or t-SNE could be explored to compare their performance with the autoencoder.

## Output

```bash
python -m autoencoder_high_dimensional_features.experiment
PyTorch version: 2.1.0.dev20230429
PyTorch built with MPS: True
MPS available: True
Using device: mps
...
Trial 22 finished with value: 0.01120121218264103 and parameters: {'learning_rate': 0.0026830263870274316, 'latent_dim': 86}. Best is trial 22 with value: 0.01120121218264103
...
Trial 49 finished with value: 42.9617805480957 and parameters: {'learning_rate': 0.00534212689449741, 'latent_dim': 71}. Best is trial 22 with value: 0.01120121218264103.
Best hyperparameters: {'learning_rate': 0.0026830263870274316, 'latent_dim': 86}
...
Epoch [10000/10000], Loss: 0.011230
Event classification without dimensionality reduction:
Dimensions before reduction: 561
Accuracy: 0.9389
Precision: 0.9404
Recall: 0.9389
F1 Score: 0.9387
Event classification with dimensionality reduction using autoencoder:
Dimensions after reduction: 86
Accuracy: 0.9257
Precision: 0.9263
Recall: 0.9257
F1 Score: 0.9253
```