Okay so,
1. Made changes to load_data.py: It now shows multiple graphs (histogram, box plots, scatter etc.)
2. Made changes to feature_engineering.py: It now outputs the HRV values in an excel file (csv). It summarises all the drowsiness values to the HRV values. (We need this file for training our model down the line).
   New code:
     1. train_model.py: Uses Random forest and trains on the given dataset (Train:70%; Test:30%)
     2. data_split.py: Creates Train;Test;Validate three way split of original csv dataset. Also equally segregates the data so it is not biased.
     3. train_model_llm: *EXPERIMENTAL* not at all necessary for our project. "Fine-tunes" LLama3.1 8B parameter model and predicts KSS value. (*Practically useless, Random Forest and XGBoost is way better and more accurate*)

Agenda:
1. Refine the train_model.py; Better implement the Random Forest model and improve upon the code. 
2. Train_dataset should have HRV values and timestamps combined for training the model. (HRV dataset needs to be trimmed down and dimensionality has to be decreased with Principal Component Analysis)
3. Implement XGBoost for model training and make a comparison chart vs Random Forest. (make new train_model.py with XGBoost)
4. Make visual representation of all our cumulative results (confusion matrix, accuracy etc.) using graphs and diagrams. (Use ChatGPT or whatever...doesn't matter).

*IMPORTANT*
  1. Before running any code, make sure you have **Python 3.11.9** installed.
  2. Install all of the packages mentioned below
    ** pip install pandas
     pip install matplotlib.pyplot
     pip install seaborn
     pip install neurokit2
     pip install os
     pip install datasets
     pip install transformers
     pip install torch**
       (Basically look at any error message "Module not found" and install that package")

   3. Change all the necessary pathways to your local system. Eg. location path of input csv, output csv. etc.
