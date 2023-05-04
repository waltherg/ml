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

Without Dimensionality Reduction:

Dimensions before reduction: 561
- Accuracy: 0.9389
- Precision: 0.9404
- Recall: 0.9389
- F1 Score: 0.9387

With Dimensionality Reduction using Autoencoder:

Dimensions after reduction: 70
- Accuracy: 0.8755
- Precision: 0.8770
- Recall: 0.8755
- F1 Score: 0.8759

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
  5%|█████████████████▏                                                                                                                                                                                                                                                                                                                       | 994/19000 [00:16<04:57, 60.49it/s]Epoch [1000/19000], Loss: 0.510228
 11%|██████████████████████████████████▍                                                                                                                                                                                                                                                                                                     | 1997/19000 [00:33<04:40, 60.52it/s]Epoch [2000/19000], Loss: 0.509706
 16%|███████████████████████████████████████████████████▊                                                                                                                                                                                                                                                                                    | 2999/19000 [00:50<04:24, 60.49it/s]Epoch [3000/19000], Loss: 0.509464
 21%|████████████████████████████████████████████████████████████████████▉                                                                                                                                                                                                                                                                   | 3995/19000 [01:06<04:07, 60.54it/s]Epoch [4000/19000], Loss: 0.509322
 26%|██████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                                                                                                                 | 4998/19000 [01:23<03:51, 60.47it/s]Epoch [5000/19000], Loss: 0.509238
 32%|███████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                                                                                                                                | 5994/19000 [01:39<03:35, 60.47it/s]Epoch [6000/19000], Loss: 0.509156
 37%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                                                                                                               | 6997/19000 [01:56<03:18, 60.45it/s]Epoch [7000/19000], Loss: 0.509234
 42%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                                                                              | 7993/19000 [02:12<03:02, 60.40it/s]Epoch [8000/19000], Loss: 0.509160
 47%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                                            | 8995/19000 [02:29<02:45, 60.50it/s]Epoch [9000/19000], Loss: 0.509106
 53%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                                           | 9998/19000 [02:45<02:28, 60.59it/s]Epoch [10000/19000], Loss: 0.508849
 58%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                         | 10997/19000 [03:02<02:12, 60.44it/s]Epoch [11000/19000], Loss: 0.508872
 63%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                        | 11993/19000 [03:18<01:55, 60.43it/s]Epoch [12000/19000], Loss: 0.508885
 68%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                       | 12995/19000 [03:35<01:39, 60.58it/s]Epoch [13000/19000], Loss: 0.508882
 74%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                      | 13999/19000 [03:51<01:22, 60.49it/s]Epoch [14000/19000], Loss: 0.508846
 79%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                    | 14999/19000 [04:08<01:06, 59.95it/s]Epoch [15000/19000], Loss: 0.508809
 84%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                   | 15999/19000 [04:25<00:49, 60.18it/s]Epoch [16000/19000], Loss: 0.508850
 89%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                  | 16993/19000 [04:41<00:33, 59.91it/s]Epoch [17000/19000], Loss: 0.508844
 95%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                 | 17997/19000 [04:58<00:16, 59.78it/s]Epoch [18000/19000], Loss: 0.508841
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉| 18995/19000 [05:15<00:00, 60.10it/s]Epoch [19000/19000], Loss: 0.508806
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 19000/19000 [05:15<00:00, 60.26it/s]
Event classification without dimensionality reduction:
Dimensions before reduction: 561
Accuracy: 0.9389
Precision: 0.9404
Recall: 0.9389
F1 Score: 0.9387
Event classification with dimensionality reduction using autoencoder:
Dimensions after reduction: 70
Accuracy: 0.8755
Precision: 0.8770
Recall: 0.8755
F1 Score: 0.8759
```