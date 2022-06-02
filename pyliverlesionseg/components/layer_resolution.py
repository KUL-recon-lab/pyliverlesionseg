from fractions import Fraction


def get_layers_scales(model, as_fractions=True, verbose=False):
    """
    Calculate the scale of all layers of a model.

    :param model:
    :param as_fractions:
    :param verbose:
    :return:
    """
    layers = model.layers
    if model.__class__.__name__ is "Sequential":
        input_layers = model._input_layers
        layers += input_layers
    else:
        input_layers = [model.get_layer(name) for name in model.input_names]
    layers_names = [l.name for l in layers]
    if verbose: 
        print(list(zip(layers_names, layers)))
    
    layers_factors = []
    for l in layers:
        keras_layer_type = l.__class__.__name__
        if keras_layer_type is "MaxPooling3D" or keras_layer_type is "Conv3D":
            layers_factors.append([Fraction(s) if as_fractions else s for s in l.strides])
        elif keras_layer_type is "UpSampling3D":
            layers_factors.append([Fraction(1, s) if as_fractions else 1 / s for s in l.size])
        else:
            layers_factors.append([Fraction(1), Fraction(1), Fraction(1)] if as_fractions else [1, 1, 1])
    if verbose: 
        print(layers_factors)
    layers_inbound_connections = []
    for i, l in enumerate(layers):
        inbound_connections = []
        for n in l._inbound_nodes:
            for _l in n.inbound_layers:
                inbound_connections.append(layers.index(_l))
        layers_inbound_connections.append(inbound_connections)
    if verbose: 
        print(layers_inbound_connections)
    layers_outbound_connections = []
    for l in layers:
        outbound_connections = []
        for n in l._outbound_nodes:
            _l = n.outbound_layer
            outbound_connections.append(layers.index(_l))
        layers_outbound_connections.append(outbound_connections)
    if verbose: 
        print(layers_outbound_connections)
            
    layers_scales = [[Fraction(0), Fraction(0), Fraction(0)] if as_fractions else [0, 0, 0] for i in range(len(layers))]
    for input_layer in input_layers:
        input_layer_idx = layers.index(input_layer)
        layers_scales[input_layer_idx] = [Fraction(1), Fraction(1), Fraction(1)] if as_fractions else [1, 1, 1]
    
    for input_layer in input_layers:
        layers_outbound_connections_ = [c for c in layers_outbound_connections]
        remaining_connections = []
        input_layer_idx = layers.index(input_layer)
        outbound_connections = [c for c in layers_outbound_connections_[input_layer_idx]]
        for outbound_connection in outbound_connections:
            remaining_connections.append([input_layer_idx, outbound_connection])
            layers_outbound_connections_[input_layer_idx].remove(outbound_connection)
        i = 0
        while remaining_connections and i < 1000000:
            inbound_connection, outbound_connection = remaining_connections.pop(0)
            inbound_scale = layers_scales[inbound_connection]
            outbound_factor = layers_factors[outbound_connection]
            outbound_scale = layers_scales[outbound_connection]
            outbound_scale = [s_in * f_out if s_in * f_out > s_out else s_out for s_in, f_out, s_out in zip(inbound_scale, outbound_factor, outbound_scale)]
            layers_scales[outbound_connection] = outbound_scale
            outbound_outbound_connections = [c for c in layers_outbound_connections_[outbound_connection]]
            for outbound_outbound_connection in outbound_outbound_connections:
                remaining_connections.append([outbound_connection, outbound_outbound_connection])
                layers_outbound_connections_[outbound_connection].remove(outbound_outbound_connection)            
            i += 1
    
    if verbose: 
        print(list(zip(layers_names, layers_scales)))
         
    for output_layer in [model.get_layer(name) for name in model.output_names]:
        layers_inbound_connections_ = [c for c in layers_inbound_connections]
        remaining_connections = []
        output_layer_idx = layers.index(output_layer)
        inbound_connections = [c for c in layers_inbound_connections_[output_layer_idx]]
        for inbound_connection in inbound_connections:
            remaining_connections.append([output_layer_idx, inbound_connection])
            layers_inbound_connections_[output_layer_idx].remove(inbound_connection)
        i = 0
        while remaining_connections and i < 1000000:
            outbound_connection, inbound_connection = remaining_connections.pop(0)
            inbound_scale = layers_scales[inbound_connection]
            outbound_factor = layers_factors[outbound_connection]
            outbound_scale = layers_scales[outbound_connection]
            inbound_scale = [s_out / f_out if s_out / f_out > s_in else s_in for s_in, f_out, s_out in zip(inbound_scale, outbound_factor, outbound_scale)]
            layers_scales[inbound_connection] = inbound_scale
            inbound_inbound_connections = [c for c in layers_inbound_connections_[inbound_connection]]
            for inbound_inbound_connection in inbound_inbound_connections:
                remaining_connections.append([inbound_connection, inbound_inbound_connection])
                layers_inbound_connections_[inbound_connection].remove(inbound_inbound_connection)            
            i += 1
    
    if verbose: 
        print("\n\n\n", list(zip(layers_names, layers_scales)))
   
    return layers_scales
    

if __name__ is "__main__":
    
    from keras.models import Model, Sequential
    from keras.layers import Input, MaxPool3D, Concatenate # MaxPool3D is just a shortcut, internally it gets the name MaxPooling3D
    from keras.layers.convolutional import Conv3D, UpSampling3D
    import pyliverlesionseg
    from pyliverlesionseg.architectures.unet_generalized import create_unet_like_model
    from pyliverlesionseg.architectures.deepmedic_generalized import create_deepmedic_like_model
    from keras import backend as K
    
    K.clear_session()
    
    model = create_unet_like_model()

    scales = get_layers_scales(model, as_fractions=True, verbose=True)
