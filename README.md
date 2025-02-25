# **Image Captioning Model 📸📝**  

## **Overview**  
This project focuses on generating descriptive captions for images using a deep learning-based image captioning model. The model extracts image features using a VGG16 pre-trained model, processes text using a tokenizer and Word2Vec embeddings, and utilizes an LSTM network to handle the sequential nature of captions. The model was trained on the Flickr8k dataset, where each image has 5 relevant captions.  

## **Dataset**  
The model is trained on the **Flickr8k dataset**, which consists of 8,000 images with 40,000 captions (5 captions per image).  

You can download the dataset from the following link:  
[**Flickr8k Dataset**](https://www.kaggle.com/datasets/adityajn105/flickr8k)

## **Methodology**  

### **1. Feature Extraction**  
Used VGG16 pre-trained model (without the fully connected layers) to extract deep features from images.  

### **2. Text Preprocessing**  
Applied a tokenizer to convert text into numerical sequences.  
Used Word2Vec embeddings to represent words in a meaningful vector space.  

### **3. Model Architecture**  
The model uses an LSTM network to handle the sequential nature of captions.  
The image features extracted from VGG16 are combined with text embeddings to generate captions.  

### **4. Training**  
The model was trained using captions where each image has 5 relevant descriptions.  
The training process optimized the model for generating relevant and meaningful captions.  

## **Evaluation**  
To measure the relevance and descriptiveness of generated captions, the model was evaluated using BLEU scores:  
- **BLEU-1 Score:** 0.498  
- **BLEU-2 Score:** 0.27  

## **Results**  
The model successfully generates contextually relevant captions for images, capturing key elements in each image.  

## **Technologies Used**  
- Python  
- TensorFlow/Keras  
- VGG16 (Pre-trained on ImageNet)  
- LSTM (Long Short-Term Memory Network)  
- Word2Vec Embeddings  
- BLEU Score for Evaluation  

  
