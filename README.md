# DeepLearning_CNN
CNN classifier for Gender, Ethnicity , Age prediction

To Do's:
---------
1. add zerroPadding proper to sliding window size
2. Adding SumPooling layer
3. Combining CSV from csvLogger and additional info at least model.json name etc
4. Somehow save number of epoches/examples that cnn has been already trained on to monitor process properly
5. scrape the page: https://www.thispersondoesnotexist.com/ -> create databest with fake faces to distinguish
6. simple django app that you can upload picture and recive predictions 
7. add transfer learnign from vgg-16 model
8. Adding early stopping criterion in callback (EarlySto

Questions that need answer:
---------------------------	
1. How handle 8mln pix new pictures which is currently a standard size / in training data we have 600x600pix pictures
	(rapid size reduction ?) 
2. Is error/backpropagation adjusting filters that are in Con2d laver ?
3. What if there are two faces on the picture and we are predicting age/gender
