# Adaptive Classification Networks

In this project, we experiment to use CNN, more specifically scattering CNN to perform unsupervised learning to predict unseen classes labels. The dataset used here is cifar. 

We mainly follows the method that is discussed in this paper: https://arxiv.org/abs/1809.06367

Our objevtive is to build a model such that it can detect K+N labels using unsupervised method after building training a model with K labels in a supervised method.

The project contains two parts:

1. Perform supervised training on K labels:

<pre>In our case, we experimented with ScatteringNet and CNN to perform supervised training on K labels.During supervised learning, we want our model to capture the features of the input image instead of memorizing them. The ability of our model to identify the features of the image is important for clustering unseen labels. 

During the training procedure, we treat any input image with class labels not in existing K labels as 1 single class. For example, when K = 5, and all the images with corresponding label not in the set of 5 labels will be grouped into a new class as our 6th label. Hence, during training, when given K class labels, the model's final layer will have dimension K + 1.

</pre>

2. Perfom unsupervised training on the new N labels, meanwhile combining the output model from step 1.

<pre>In our case, we used Gaussian Mixture Model to perform unsupervised training on the new N labels.

From step 1, we have built a model to detect the existing K class labels from unseen labels. In this step, we use GMM to cluster all the images classified as "unseen label" into N different clusters.
</pre>

# 5 Unknown Classes = \[5,6,7,8,9\]
Firstly, we set the number of untargeted class to be 5 (5,6,7,8,9). Hence, we will train our Scattering CNN with 6 classes (0,1,2,3,4 and unknown class), and based on the results we will combine this with GMM to help us perform unsupervised training. Below is the Scattering CNN training result:

![Screen Shot 2022-01-06 at 12 09 45 PM](https://user-images.githubusercontent.com/54965707/148422179-6d0de2f0-6b61-4afb-acaa-b9a2b245a9d9.png)

After training the Scattering CNN, we tried to use single GMM and multiple GMMs to perform clustering. The below result is for single GMM:

![Screen Shot 2022-01-06 at 12 10 59 PM](https://user-images.githubusercontent.com/54965707/148422328-76fc462e-46cd-411c-868c-74ffc14e9c6f.png)

The below result is for multiple GMMs:

![Screen Shot 2022-01-06 at 12 11 24 PM](https://user-images.githubusercontent.com/54965707/148422394-aa3fb16d-aef3-44bd-b00a-813ccf3f565a.png)

# 4 Unknown Classes = \[6,7,8,9\]

Secondly, we set the number of untargeted class to be 4 (6,7,8,9). Hence, we will train our Scattering CNN with 7 classes (0,1,2,3,4,5 and unknown class), and based on the results we will combine this with GMM to help us perform unsupervised training. Below is the Scattering CNN training result:

![Screen Shot 2022-01-06 at 12 12 57 PM](https://user-images.githubusercontent.com/54965707/148422650-5a4ecf40-3593-4673-a78c-b6957621634e.png)

After training the Scattering CNN, we tried to use single GMM and multiple GMMs to perform clustering. The below result is for single GMM:

![Screen Shot 2022-01-06 at 12 13 12 PM](https://user-images.githubusercontent.com/54965707/148422692-904fcf41-e181-4b30-a01b-cfe2063a9b33.png)

The below result is for multiple GMMs:

![Screen Shot 2022-01-06 at 12 13 30 PM](https://user-images.githubusercontent.com/54965707/148422740-6ca539a3-c15f-4c59-90cb-508532e433ae.png)

# 3 Unknown Classes = \[7,8,9\]

Lastly, we set the number of untargeted class to be 3 (7,8,9). Hence, we will train our Scattering CNN with 8 classes (0,1,2,3,4,5,6 and unknown class), and based on the results we will combine this with GMM to help us perform unsupervised training. Below is the Scattering CNN training result:

![Screen Shot 2022-01-06 at 12 13 52 PM](https://user-images.githubusercontent.com/54965707/148422802-2476ca5e-2121-450a-81b3-68e38de81656.png)

After training the Scattering CNN, we tried to use single GMM and multiple GMMs to perform clustering. The below result is for single GMM:

![Screen Shot 2022-01-06 at 12 14 40 PM](https://user-images.githubusercontent.com/54965707/148422917-0973c0be-f11f-4169-b025-2c7613d5d8ee.png)

The below result is for multiple GMMs:

![Screen Shot 2022-01-06 at 12 14 51 PM](https://user-images.githubusercontent.com/54965707/148422944-0e7ec75d-c3ee-4b53-9ebd-0043beaf4d93.png)

# Conclusion

Our ScatterCNN + GMM network makes fairly well prediction as the average among all the classes for all the experiments are above 80%. We experiment on directly applying GMM to the images but it has a poor accuracy, which infers our scatterCNN does a good job at feature extraction. However we could observe that the accuracy of pretrained-model (ScatterCNN) drops as fewer number of classes is feed in the first stage. Furthermore, different implementation of GMM classifiers in the unsupervised stage don't make huge difference at accuracy.

While we are satisfied with the performance of scatter layer, our CNN's structure is too simple, this could be the reason that few classes involved in first stage don't have reasonable accuracy. To solve this problem we could switch our network with ResNet, VGG and other well-known architectures, we also expect that by keeping parameters in those trained models or set a very small learning rate for those models, we could boost our accuracy without spending much more training time by the power of transfer learning. <br>
We assume the issue could also caused by imbalanced classificatoin. I.E the seen classes has a average class size: S, but the unseen group always have a bigger size as it is the combination of 2 or more classes. It leads to the unequal distribution of classes. The easist solution will be resampling our dataset to produce more balanced classes. <br>
