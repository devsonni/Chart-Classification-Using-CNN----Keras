# Convolution Nueral Network For Classification Of Charts   
## Task 1 & 2 : Image Sorting & Result of Trained Models    
### Image sorting              
        
All charts transfered to Train/(spesific folder) as their labels and randomly 20% images transfered into validation folder. Also, images separated in test/(particular) folder, checkout mentioned structure.        
This will make labling process much easier with the help of ImageDataGenerator function.    
     
<img align="center" height="400" width="350" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/charts/Folder%20Structure.jpg">      
         
        
### Model Results      
    
Both Models trained for 10 epochs.        
Model 1: 2, 2D Conl layers with 16- 3X3 & 32-3X3 filters, each has a MaxPooling layer of 2X2, got 97% accuracy for validation set, and 96% for test set. [Saved Model](https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/charts/16-32-Arch-V1.h5), [Script](https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/charts/ConvolNN_V1.py)     
         
<img align="left" height="450" width="350" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/V1/4th/Accuracy.jpeg">
<img align="right" height="450" width="350" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/V1/4th/Loss.jpeg">     
<img align="center" height="450" width="550" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/V1/4th/Confusion%20Mat.jpeg">     
        
Model 2: 2, 2D Conl layers with 32-3X3 & 64-3X3 filters, each has a MaxPooling layer of 2X2, got 98% accuracy for validation set, and 98% for test set. [Saved Model](https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/charts/32-64-Arch-V1.h5), [Script](https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/charts/ConvolNN_V2.py)        
     
<img align="left" height="450" width="350" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/V2/1st/Accuracy.jpeg">
<img align="right" height="450" width="350" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/V2/1st/Loss.jpeg">     
<img align="center" height="450" width="550" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/V2/1st/Confusion.jpeg">      
      
CNNs works same as humas brains they did mistake in recognizing doted line and line, I think that was most difficult for them.     
     

## Task 3: Fine Tuning of Pretrained Models    
     
Updated VGG16 pretrained award winning model. Where, I changed last 1000 layer prediction to 5 prediction and froze all the weights insted of last layer's weights and trained model for just 5 epochs, and got 100% accuracy for both validation and test sets.      
   
      
<img align="left" height="450" width="350" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/VGG16UP/Accuracy.jpeg">
<img align="right" height="450" width="350" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/VGG16UP/Loss.jpeg">     
<img align="center" height="450" width="550" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/VGG16UP/Confusion%20Matrix.jpeg">    

      
## Task 4: GradCAM     
       
### Dot Line      

<img align="left" height="450" width="700" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/GradCAM/D%20Line%201.jpeg">
<img align="right" height="450" width="700" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/GradCAM/D%20Line%202.jpeg">      
     
### Line       
     
<img align="left" height="450" width="700" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/GradCAM/Line%201.jpeg">
<img align="right" height="450" width="700" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/GradCAM/Line%202.jpeg">     
      
### Vertical Bar    
      
<img align="left" height="450" width="700" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/GradCAM/V%20Bar%201.jpeg">
<img align="right" height="450" width="700" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/GradCAM/V%20Bar%202.jpeg">     
    
### Horizontal Bar      
      
<img align="left" height="450" width="700" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/GradCAM/H%20Bar%201.jpeg">
<img align="right" height="450" width="700" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/GradCAM/H%20Bar%202.jpeg">     
       
### Pie     
     
<img align="left" height="450" width="700" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/GradCAM/Pie%201.jpeg">
<img align="right" height="450" width="700" src="https://github.com/devsonni/Chart-Classification-Using-CNN----Keras/blob/main/Results/GradCAM/Pie%202.jpeg">     


