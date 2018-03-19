# Emotion Reciprocation for Improving Mental Well Being

## Description 
Stress, anxiety, depression, anger and pressure are the emotional situations which most humans face when they are going through difficult times. Happiness, calmness, serenity, pleasure and excitement are occasions where humans benefit from since these emotional situations are associated with a positive mental health. During both these contrasting occasions, humans often seek to share their emotional state with others in a socially dynamic environment. This can be seen as a type of relationship between the subject and one or more characters. Research in psychology has shown that the presence of an external entity plays a vital role during emotional transitions, and subjects in relationships which involve reciprocation of emotional states, benefit the most with respect to how fast the subject converges to a positive mental state. Our project is inspired by the above and we present a novel system to aid in emotional state transition by reciprocating his/her emotional state through sensory based stimulation. A
generic framework of our system is detailed below:  
  
### 1. Identifying the "Trigger" through Emotional State Recognition:  
This can be achieved through one or more of the below subtasks:  
	a. Facial expression recognition  
	b. Pulse Analysis  
	c. Breathing Pattern Analysis  
	d. Brain-wave Analysis  
	e. Body Heat Analysis  
  
### 2. Determining Individual Characteristics for Delivery of Appropriate Response.  
Examples of individual characteristics and the ways to collect them are:  
	a. Medication information through prescriptions and medical reports  
	b. Emotional Volatility analysis using surveys  

### 3. Delivery of Appropriate Actions or Responses.  
Some appropriate responses can be (not limited to):  
	a. Music therapy  
	b. Visual Therapy  
	c. Medication notification/reminder  
	d. Notification to concerned medical practitioners  
	e. Rational decision-making assistant  
	f. Logging Progress / Emotional State Statistics  
  
## Scope of Project:  
We limit the scope of this project to developing the trigger of Facial Expression Recognition (FER) (1a) and deliver the responses of Music Therapy (3a), Emotional State Statistics (3f) and if time permits, Visual Therapy (3b).

## Facial Expression Recognition: Video Classification using CNN - RNN Pipeline  
Datasets Used:  

1. Oulu CASIA NIR & VIS Facial Expression Database:  
G. Zhao, X. Huang, M. Taini, S.Z. Li & M. Pietikäinen (2011): Facial expression
recognition from near-infrared videos. Image and Vision Computing,
29(9):607-619.  

### References:
1. The little "bird" which helped us to go in the right direction:  
https://blog.coast.ai/continuous-online-video-classification-with-tensorflow-inception-and-a-raspberry-pi-785c8b1e13e1
  
Documentation will be updated soon  
 
## Music Emotion Recognition: Audio Spectrogram Classification using LRCN
  
Dataset Used: MediaEval's Database for Emotional Analysis in Music (DEAM)  
  
Aljanaki A, Yang Y-H, Soleymani M (2017) Developing a benchmark for emotional analysis of music. PLoS ONE 12(3): e0173392. https://doi.org/10.1371/journal.pone.0173392  


Documentation will be updates soon  

### References:
1. The best explanation I have read on choosing CNN filter sizes for spectrograms by Jordi Pons:  
http://www.jordipons.me/cnn-filter-shapes-discussion/  

2. Different approach but less explicitly explained - still useful:  
http://deepsound.io/music_genre_recognition.html  

3. An excellent visual review of the methods:  
A Tutorial on Deep Learning for Music Information Retrieval - Choi et al.  
https://arxiv.org/pdf/1709.04396.pdf
