# Deep Learning - Image Classification Problem

--------------------------------------------------------------------------------
## Project Objective: age estimation, ethnicity & gender classification
--------------------------------------------------------------------------------
## Dataset: https://susanqq.github.io/UTKFace/?fbclid=IwAR29u0ed1nMmRs8WI2XwiMUsGuUFh6yuEexYsXDmV2NeMoInYYvnPVmy0Uk
--------------------------------------------------------------------------------

Question to ask:
1. Is the scope of the project big enough/difficult enough
2. Expert knowledge: it is possible to train CNN enough in ~1month/comparing different approaches etc.
	Speed of training is ~5 example / second 
3. Can it be worth to combine few architecure for examlple to have comon first few layers and differ architecutre only on fully conected layers to distinguish different features ?
	It is even possible ?
4. 

# Goals of the project:
--------------------------------------------------------------------------------
I) Performing accurate prediction on unseen data:
	1) Create small app where you can load your own image and perform classification
	2) Looking into limitations like min or max resolution, % of face on picture,
	   % of full picture being face vs accuracy of prediction
II) Comapring different approaches to the problem:
	1) classifing everything with one architecture ( 2 x 4 x 116 classes)
	2) combing two CNN ( one CNN for gender and race + one CNN for Age )
        based on complexity of NN, time of training, avg error etc
III) Addictional task: ( If we have time )
	1) comapring ACC of different architecture
		( trade off complexity vs accuracy )
		( minimal complexity for given treshold of ACC )
	2) comapring our approach with pre-traind open-source model
	3) Impact of filtered pictures on age predictions
IV) Extra bonus task
	1) Face detection -> find face on picture and mark it with square around it
	
	

# USEFUL READINGS:
--------------------------------------------------------------------------------

Blog Posts about Image Classification
--------------------------------------------------------------------------------
https://towardsdatascience.com/wtf-is-image-classification-8e78a8235acb
https://medium.freecodecamp.org/how-to-build-the-best-image-classifier-3c72010b3d55
https://towardsdatascience.com/easy-image-classification-with-tensorflow-2-0-f734fee52d13

GENERAL CNN KNOWLEDGE: 
A Beginner's Guide To Understanding Convolutional Neural Networks (Part 1, 2 and 3)
------------------------------------------------------------
https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/
https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/
https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html

Gentle Dive into Math Behind Convolutional Neural Networks
------------------------------------------------------------
https://towardsdatascience.com/gentle-dive-into-math-behind-convolutional-neural-networks-79a07dd44cf9

Keras ImageDataGenerator methods: An easy guide
------------------------------------------------------------
https://medium.com/datadriveninvestor/keras-imagedatagenerator-methods-an-easy-guide-550ecd3c0a92

TO BE USED AS A BENCHMARK -> pretraind CNN model with VGG-16 architecture
------------------------------------------------------------
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/




USELESS READING and Papers (but good for references)
=================================================================================
http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

A Friendly Introduction to Cross-Entropy Loss
------------------------------------------------------------
https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/

Softmax classification with cross-entropy:
------------------------------------------------------------
https://peterroelants.github.io/posts/cross-entropy-softmax/



DeepLearning_CNN
CNN classifier for Gender, Ethnicity , Age prediction

To Do's:
---------
4. Somehow save number of epoches/examples that cnn has been already trained on to monitor process properly - DOMINKA
1. add zerroPadding proper to sliding window size - DOMINKA
8. Create CNN instance for age regression prediction - DOMINKA

3. Prepare input for multioutput classification (preferably python if no PowerShell/bash) - LUCAS
10. Deal with file paths

5.1 Write automatic hyperparameter tunning with grid search - JOANA
5.2 Compe up with ideas for bencharmking models (for example: same number of epochs, same early stopping criteris etc) - JOANA
5.3 Saving benchamrking and code for plotting - JOANA

7. add transfer learnign from vgg-16 model
2. Adding SumPooling layer
6. simple django app that you can upload picture and recive predictions 
5. scrape the page: https://www.thispersondoesnotexist.com/ -> create databest with fake faces to distinguish

Questions that need answer:
---------------------------	
1. How handle 8mln pix new pictures which is currently a standard size / in training data we have 600x600pix pictures
	(rapid size reduction ?) 
2. Is error/backpropagation adjusting filters that are in Con2d laver ?
3. What if there are two faces on the picture and we are predicting age/gender
