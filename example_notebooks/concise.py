
import tensorflow as tf

import segnet.metrics as mts

from segnet.models import unet
from segnet.models import multiresunet as mru
from segnet.models import multiresunet2 as mru2
from segnet.models import multiresunet3 as mru3
from segnet.utils.Segmed import Segmed

def main():

    dataset_paths = {
        "isbi":          "/Users/gml/Google Drive/DCI-Net/Colab_data/ISBI_neural/structured", 
        "colonoscopy":   "/Users/gml/Google Drive/DCI-Net/Colab_data/colonoscopy",   # Full original
        "dermoscopy80":  "/Users/gml/Google Drive/DCI-Net/Colab_data/dermoscopy80",  # reduced to 80 images
        "dermoscopy150": "/Users/gml/Google Drive/DCI-Net/Colab_data/dermoscopy150", # reduced to 150, 
        "chinese1":      "/Users/gml/Google Drive/DCI-Net/Colab_data/Dataset 2"      # Chinese dataset
    }

    optimizers = {
        "chinese": tf.keras.optimizers.SGD(learning_rate=0.06, momentum=0.2, nesterov=False),
        "Original Adam": tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=10e-8)
    }

    my_compiling_kw = {
        'optimizer': optimizers["Original Adam"],
        'loss': 'binary_crossentropy',
        'metrics': [
          mts.jaccard_index, mts.dice_coef,
          mts.O_Rate, mts.U_Rate, mts.Err_rate
        ]
    }

    my_hyper_params = {
        'batch_size': 25,
        'epochs': 150,
        'steps_per_epoch': 6
    }

    models = {
        "Unet": unet(),
        "MultiResUNet Edwin": mru.MultiResUnet(),
        "MultiResUNet Gustavo": mru2.MultiResUNet(),
        "MultiResUNet Original": mru3.MultiResUnet()
    }

    model = "Unet" # see models.keys()

    x = Segmed(
        model = models[model],
        name = model, 
        base_dir = "/Users/gml/Google Drive/DCI-Net/SegMedLogs/LocalTests", 
        data_path = dataset_paths["dermoscopy80"], 
        author = "Gustavo Magaña"
    )

    x.train(
        compiling_kw=my_compiling_kw, 
        hyper_params=my_hyper_params
    )


if __name__ == "__main__":
    main()

