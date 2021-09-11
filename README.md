# CS6300-Speech-Technology
## Project 1: 
This project primarily deals with text dependent speaker verification on a closed set data. The dataset contains utterances from 23 speakers and each speaker has about 5 instances out of which 3 of them are being taken as templates and 2 of them as test instances. The implementation of the verification will involve the use of Dynamic Time Warping, a dynamic programming technique which is a robust technique to compare sequences of variable length. All the utterances were recorded at a sampling frequency of 16kHz and were single channel recordings.
## Project 2: 
The objective of this project is to build a Vanilla Gaussian Mixture Model system to do the task of text independent speaker identification. The dataset contains
200 speakers and is taken from the TIMIT/NTIMIT databases. Before training the GMMs, MFCC features were extracted from both the TIMID and NTIMID data. Features were extracted at 100 frames per second. Both MFC and Delta Coefficients were extracted from the data. To reduce the influence of noise Voice Activity Detection (VAD) was used.
## Project 3:
This project involves building Hidden Markov Models for isolated digit recognition followed by continuous digit recognition using the isolated digit HMM through concatenation.
## Final Project:
This project involves the use of End to End Encoder-Decoder models to facilitate the task of classifying speech files of isolated digits into the corresponding classes. We employed the used of PyTorch Deep Learning framework for purposes of coding. The models which have been trained have been with and without **Attention mechanisms**, and the results obtained were compared. The Speech files were converted into MFCC features with each feature having a dimensionality of 38.
