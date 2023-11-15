# DIFFUSION MODELS

This is a repository that re-implements the DDPMs model based on the notebook at [PyTorch implementation of 'Denoising Diffusion Probabilistic Models'](https://github.com/awjuliani/pytorch-diffusion).

The components of the U-Net network that I coded can be found in the directory `src/models/modules/`. The implementation of the forward diffusion and reverse diffusion algorithms can be found in the file `src/models/diffusion_model.py`.

To rerun the experiment, first enter your Wandb API Key in the `run.sh` file for experiment tracking. Then, change the value of `experiment` to 'mnist' if you want to run the experiment with the MNIST dataset, or 'cifar' if you want to run the experiment with the CIFAR dataset. Finally, execute the command below:

```bash
bash run.sh
```

## Results

### MNIST

![](/images/mnist.png)

### CIFAR

![](/images/cifar.png)
