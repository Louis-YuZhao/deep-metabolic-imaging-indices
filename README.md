# deep-metabolic-imaging-biomarker
Code for paper titled "Differential diagnosis of parkinsonism with deep metabolic imaging biomarker" 

by Yu Zhao, University of Bern and Technical University of Munich 

last modified 07.21.2020

# Requirement:
  > Python 2.7.3  
  > tensorflow 1.9.0  
  > Keras 2.2.2  
  > keras-contrib 2.0.8  
  > pandas 0.24.2  
  > scikit-image 0.14.0  
  > scikit-learn 0.19.2  
  > SimpleITK 1.1.0  


# Guideline for utilizing:

(1) Data preparing

    python prepareData.py

(2) trainging the model:
    
    python train.py

(3) Evaluated the performance of the trained model:

    python evaluate.py
