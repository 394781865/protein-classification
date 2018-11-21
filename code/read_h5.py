import h5py
import numpy as np

def load_weights(filepath):
    '''
        This method does not make use of Sequential.set_weights()
        for backwards compatibility.
    '''
    # Loads weights from HDF5 file
    import h5py
    f = h5py.File(filepath)
    for k in range(f.attrs['nb_layers']):
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        print(weights)
    f.close()

def get_weights(weight_file_path, layer_name):
    f = h5py.File(weight_file_path)  # 读取weights h5文件返回File类
    try:
        #if len(f.attrs.items()):
        #    print("{} contains: ".format(weight_file_path))
        #    print("Root attributes:")

        def printname(name):
            print(name)

        #f.visit(printname)
        g = f[layer_name]
        val = g.value
        #print(val.shape)
        #print(val)

        if 'b' in layer_name.split('_'):
            return val

        #new_init = np.zeros([7,7,1,64])
        new_init = val[:,:,:1,:]
        new_val = np.concatenate((val, new_init), axis=2)
        #print(new_val.shape)
        #print(new_val[1][1][:][:])
        return new_val

    except:
        f.close()

if __name__ == '__main__':
    #path = '../pretrained_model/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'
    path = '../pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    #layer_name = 'conv1/conv/conv1/conv_1/kernel:0'
    layer_name = 'conv1/conv1_W_1:0'
    get_weights(path, layer_name)
