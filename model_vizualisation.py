import matplotlib.pyplot as plt
from tnesorflow.keras.models import load_model
from keras.utils.layer_utils import count_params

def visualize_model_with_dropout(model):
    layer_names = [layer.__class__.__name__ for layer in model.layers]
    layer_params = [count_params(layer.weights) for layer in model.layers]
    
    # Figure setup
    plt.figure(figsize=(10, 6))
    colors = {'Dense': 'skyblue', 'Dropout': 'salmon'}
    
    for i, (name, params) in enumerate(zip(layer_names, layer_params)):
        color = colors.get(name, 'gray')
        plt.barh(i, params, color=color)
        plt.text(params, i, f'{name} ({params} params)', va='center')
    
    plt.xlabel("Number of Parameters")
    plt.ylabel("Layers")
    plt.title("Model Architecture with Dropout Layers Highlighted")
    plt.show()

visualize_model_with_dropout(model)
