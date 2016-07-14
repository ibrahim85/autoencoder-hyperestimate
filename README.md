## Autoencoder-hyperestimate

Generate deep neural auto-encoder based on convolution and max-pool layers to reconstruct images. Modify hyper-parameters easily, such as number of layers, amount of filters, ...

This project is based on Caffe, and runs thanks to Nut.

### How To
First define the parameters of your architecture in `main()` of `src/main.py`, such as:
```python
# Default architecture of the paper
settings = {
    "featuresPerLayer": 64,
    "convLayersPerBlock": 2,
    "blocks": 3,
    "kernelSize": 3,
    "kernelSizeLocal": 1,
    "strideConv": 1,
    "stridePool": 2,
    "inputSize": 32
    }
```
Then run the command `nut hyp` to generate the Caffe prototxt files for pretraining and training phases. This will also output the command which you should run to perform all the training. This will also include the generation of images to compare original images of the training set and reconstructed images by the network:

Full command:
```terminal
$ nut hyp
libdc1394 error: Failed to initialize libdc1394

[...]

Full command:
nut --exec='cd /src; time caffe train --solver=prototxt/16-07-14_15h39m04s_0_00_pretrainingConvCifar10_solver.sh > 16-07-14_15h39m04s_0_00_logs.txt 2>&1; time caffe train --solver=prototxt/16-07-14_15h39m04s_0_10_pretrainClassificationFrozen_solver.sh --weights=snapshots/16-07-14_15h39m04s_0_00_pretrainingConvCifar10_iter_2000.caffemodel > 16-07-14_15h39m04s_0_10_logs.txt 2>&1; time caffe train --solver=prototxt/16-07-14_15h39m04s_0_11_pretrainClassification_solver.sh --weights=snapshots/16-07-14_15h39m04s_0_10_pretrainClassificationFrozen_iter_3000.caffemodel > 16-07-14_15h39m04s_0_11_logs.txt 2>&1; time caffe train --solver=prototxt/16-07-14_15h39m04s_0_20_reconstructIncremental1_solver.sh --weights=snapshots/16-07-14_15h39m04s_0_11_pretrainClassification_iter_20000.caffemodel > 16-07-14_15h39m04s_0_20_logs.txt 2>&1; time caffe train --solver=prototxt/16-07-14_15h39m04s_0_40_reconstructFull_solver.sh --weights=snapshots/16-07-14_15h39m04s_0_20_reconstructIncremental1_iter_2000.caffemodel > 16-07-14_15h39m04s_0_40_logs.txt 2>&1; time caffe test --weights=snapshots/16-07-14_15h39m04s_0_40_reconstructFull_iter_30000.caffemodel --model=prototxt/16-07-14_15h39m04s_0_40_reconstructFull_net.sh --gpu=0 > 16-07-14_15h39m04s_0_40_logs_tests.txt 2>&1; time python2 VisualizeReconstructionOfLayer.py snapshots/16-07-14_15h39m04s_0_40_reconstructFull_iter_30000.caffemodel prototxt/16-07-14_15h39m04s_0_40_reconstructFull_net.sh; time caffe train --solver=prototxt/16-07-14_15h39m04s_0_50_reconstructFullFC0_solver.sh --weights=snapshots/16-07-14_15h39m04s_0_40_reconstructFull_iter_30000.caffemodel > 16-07-14_15h39m04s_0_50_logs.txt 2>&1; time caffe test --weights=snapshots/16-07-14_15h39m04s_0_50_reconstructFullFC0_iter_30000.caffemodel --model=prototxt/16-07-14_15h39m04s_0_50_reconstructFullFC0_net.sh --gpu=0 > 16-07-14_15h39m04s_0_50_logs_tests.txt 2>&1; time python2 VisualizeReconstructionOfLayer.py snapshots/16-07-14_15h39m04s_0_50_reconstructFullFC0_iter_30000.caffemodel prototxt/16-07-14_15h39m04s_0_50_reconstructFullFC0_net.sh; time caffe train --solver=prototxt/16-07-14_15h39m04s_0_60_reconstructFullFC0unfrozen_solver.sh --weights=snapshots/16-07-14_15h39m04s_0_50_reconstructFullFC0_iter_30000.caffemodel > 16-07-14_15h39m04s_0_60_logs.txt 2>&1; time caffe test --weights=snapshots/16-07-14_15h39m04s_0_60_reconstructFullFC0unfrozen_iter_30000.caffemodel --model=prototxt/16-07-14_15h39m04s_0_60_reconstructFullFC0unfrozen_net.sh --gpu=0 > 16-07-14_15h39m04s_0_60_logs_tests.txt 2>&1; python2 VisualizeReconstructionOfLayer.py snapshots/16-07-14_15h39m04s_0_60_reconstructFullFC0unfrozen_iter_30000.caffemodel prototxt/16-07-14_15h39m04s_0_60_reconstructFullFC0unfrozen_net.sh'
```

You can either copy-paste this command to your terminal, are run `16-07-14_15h39m04s_0_run.sh`, which contains the same command. Generated prototxt files are in `src/prototxt`. Snapshots will be saved in `src/snapshots`, and all the logs of Caffe will be saved to files in `logs/`.

### Visualize Reconstruction On The Test Set
This project relies on an old version of Caffe, so specifying the test/training set is not possible with Pycaffe. As a result, visuals are generated from the training set. There is usually no difference between reconstruction on the train and test set, due to the variety of images. But if you wish to compare with test set, you may re-run the `src/VisualizeReconstructionOfLayer.py` script after editing the path to the training set in the prototxt files (make the train set point to the test set).

Otherwise, you may use the branch [visutestset](https://github.com/matthieudelaro/autoencoder-hyperestimate/tree/visutestset) to generate both train set and test set images during the training. However this branch had not been well tested, so be warn of potential weird results.

### Limitation / Improvements
- The dataset is not zero-centered. The pixel colors of the images of the dataset are encoded from 0 to 255. So it would be better to specify a transformation in the data layer to fix this. Variance should be handled as well.
- ReLU activationn use in this project is probably too big. It leads the network to consider a local minimum, which corresponds to the average image to reconstruct. Reducing its value (and adjusting other hyperparameters accordingly) should speedup the training, and improve the results.
- This project doesn't generate plots with learning rate and loss. It would be a nice feature, and it shoud be easy to add thanks to the extra tools of Caffe.


### Cite
This project has been created to run experiments for the paper *Tunnel Effect in CNNs (2016)*. Please cite this paper if you use this project in your research. 
