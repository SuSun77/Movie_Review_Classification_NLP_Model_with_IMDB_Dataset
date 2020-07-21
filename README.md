# IMDB   
RNN model with preprocessing on the original IMDB dataset.   
Data can be downloaded from http://ai.stanford.edu/~amaas/data/sentiment/    
      
The architecture of the model:    
Embedding(input_dim=20000, output_dim=32, input_length=100)   
Flatten()    
Dense(units=256, activation='relu')   
Dense(units=1, activation='sigmoid')    
     
Folder structure:   
* data   
  * aclImdb   
    * test   
      * pos   
      * neg   
    * train   
      * pos   
      * neg   
* models   
  * saved model   
* test_NLP.py   
* train_NLP.py   
      
    
