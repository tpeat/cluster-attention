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
