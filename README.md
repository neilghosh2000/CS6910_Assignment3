# CS6910_Assignment3

The description of the different files in the repository are as follows:

1. best_model - Contains the saved weights of the vanilla model trained in tensorflow (36.89% test accuracy).
2. models - Contains the saved weights of the models trained in pytorch  
    + model_pytorch_AT.pt, model_pytorch_AT_state.pt - Saved weights of the model with attention (30.91% test accuracy)
    + model_pytorch_noAT.pt, model_pytorch_noAT_state.pt - Saved weights of the model without attention (6.88% test accuracy)  
3. predictions_attention_pytorch - Contains images of the heatmap, connectivity visualisation, sample predictions and an excel file of all the predictions of the test data made by the pytorch attention model.
4. predictions_vanilla_pytorch - Contains images of the sample predictions and an excel file of all the predictions of the test data made by the pytorch vanilla model.
5. predictions_vanilla - Contains images of the sample predictions, characterwise errors and an excel file of all the predictions of the test data made by the tensorflow vanilla model.
6. PyTorch_Vanilla.ipynb - .ipynb file for the PyTorch vanilla model.
7. PyTorch_Attention.ipynb - .ipynb file for the PyTorch attention model.
8. Seq2Seq_Vanilla.ipynb - .ipynb file for the Tensorflow vanilla model.
9. sweep.yaml - Contains the hyperparameter values to run the sweep for the Tensorflow vanilla model.
10. train.py - Similar to Seq2Seq_Vanilla.ipynb but in .py format to run the sweep on wandb.

Best results - 
   + Tensorflow Vanilla - Characterwise Accuracy = 0.9518, Train Loss = 0.2144, Wordwise Accuracy = 0.3689
   + PyTorch Vanilla - Wordwise Accuracy = 0.0688
   + PyTorch Attention - Wordwise Accuracy = 0.3091

Link to Wandb report - https://wandb.ai/arnesh_neil/CS6910_Assignment_3/reports/CS6910-Assignment-3-Report--Vmlldzo2NTU4ODg

