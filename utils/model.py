
import network

def load_model_arch(model_name, num_classes):

    try:
        model_func = getattr(network, model_name) 
    except:
        raise ValueError('Invalid model name : {}'.format(model_name))
    
    model = model_func(classes = num_classes)
    return model