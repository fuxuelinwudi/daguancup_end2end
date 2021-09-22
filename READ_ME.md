#The 5th Dagan Cup, Team name: XiaoChuan Sun , 4th in the A list, 7th in the B list, single model throughout.

# 1.data process details

##### 1.1.The maximum length of a sentence is limited to 128, and any sentence longer than 128 is truncated (by taking the first 32 and the last 96).


# 2.pretrain details

##### 2.1.The data used is the first 18W json (title+content) of the unlabeled data, totaling 36W (training set and test set data are not used, because I forgot to use them).

##### 2.2.The pre-training model used is nezha-cn-base, and the pre-training task is albert's ngram mask, as well as the Word Structural Objective task borrowed from structbert, in the time of mask, a randomly selected trigram is disrupted, and while the model predicts the original token, it also does the restoration operation, which is equivalent to the improvement of this task of structbert.


# 3.finetune details

##### 3.1.Regular tricks are: PGD, Lookahead, EMA, stratified learning rate, TSA, etc.
##### 3.2.Customized the model architecture as follows.
###### 3.2.1.Taking the CLS of the last five layers of all hidden layer states for splicing works best (tried many kinds of structures, such as: post-connected CNN/LSTM, MSD, MEAN-POOLING, etc.).
###### 3.2.2.Because the data comes with two levels of labels, the labels are cut (primary label: 10, secondary label 35) and the loss is calculated separately (for the output hidden_state, it goes through two linear layers respectively, each linear layer output dimension corresponds to a different number of labels).
###### 3.2.3.The self-researched method, in the model fine-tuning, in each batch, let the model to predict the training set, the prediction results and the real label between the loss of feedback, pulling the distance between the predicted label and the real label, the effect has slightly improved, in other data sets have been tested, not deep investigation of it, is an innovative point.
