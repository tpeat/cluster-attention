# Experiment 2: CHAI
Experiment inspired by CHAI paper: https://arxiv.org/abs/2403.08058. Involves clustering attention heads post training to improve space savings for model inference.

### Files

- **`clustering.py`**  
  Contains functions for:
  - Creating clusters of attention heads.
  - Assigning weights to these clusters.

- **`eval_model.py`**  
  Provides utilities for evaluating the model's performance after clustering attention heasds.
  - Note: change clustered_weights_path to clustered local checkpoint path 

- **`main.py`**  
  Entry point for the experiment.  
  - Clusters attention weights.
  - Saves the modified model checkpoint.
  - Note: change pretrained_model_path to  local pretrained checkpoint path 

- **`model.py`**  
  Implements the `load_pretrained_model` function to load the pre-trained model and tokenizer for processing.
   

- **`plot_results.py`**  
  Includes functions for:
  - Plotting specific attention head.
  - Plotting entire layer and assigned clusters.


### Run experiment
1. Select number of clusters for attention heads
2. Identify path to save model
3. Run ```python main.py```
4. Run ```python eval_model.py``` to generate bleu score
   
