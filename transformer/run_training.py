from model_training import load_trained_model, ModelHyperParams, TrainingHyperParams
import torch

def main():
    print("Initializing Training...")
    device = torch.device(
        'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print("Device: ", device)
    training_hp = TrainingHyperParams()
    model_hp = ModelHyperParams()
    print('Training Hyperparameters:', training_hp)
    print('---------------------------------------------------------------------------')
    print('Model Hyperparameters:', model_hp)
    
    if device == 'cuda':
        torch.cuda.empty_cache()
    training_hp = TrainingHyperParams()
    model_hp = ModelHyperParams()
    model = load_trained_model(training_hp, model_hp)
    
    
if __name__ == '__main__':
    main()