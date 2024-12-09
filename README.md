# Cluster Attention

We hypothesize that over time gradient descent and self-attention work together
to push similar objects in x (tokens input to transformer block) together, thereby forming clusters and that
there is alternative architecture component that can more effectively find these
clusters without repeated $O(n^2)$ operations.

[Full report](NLP_Project.pdf)

# Experiment 1: Token Merging

In this experiment, we tested the impact of merging tokens with similar information in the encoder block. We modified Google's t5 base transformer. All experimental files are located in the TokenMerging folder. 

To run the experiment, either:
* Run the train_model.py file
* Run the first experiment under the NLP.ipynb Jupyter notebook


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
   
