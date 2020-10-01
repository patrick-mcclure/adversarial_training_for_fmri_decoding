# Adversarial Training for fMRI Decoding from Beta Volumes

## Train

Below is a command-line call to train a 3D convolutional neural netwok (CNN) model to perform multinomial decoding of fMRI beta volumes. Explaination of the command line arguements are also below.

```
taskandyeshallperceive train \
  --model-dir={modeldirpath}
  --input-csv={inputcsvpath}
  --n-classes=7 \
  --n-m=4 \
  --batch-size=8 \
  --n-epochs=50
```

- `--model-dir`: the directory in which to save model checkpoints.
- `--input-csv`: path to CSV file with a row for each input with the first column containing the file path to a beta volume nifti file, the secind column containing a subject ID, and the third column containing the numerical code for the decoding target. The file must have a header, although the column names can be arbitrary.
- `--n-classes`: the number of classes being decoded.
- `--n-m`: the number of adversarial training steps to take for each batch. The default is 1, which will disable adversarial training.
- `--batch-size`: the number of input examples per batch.
- `--n-epochs`: the number of input examples per batch.



## Predict

Below is a command-line call to predict the decoding targets for fMRI beta volumes using a 3D CNN model and to generate input gradient maps for a given target. Explaination of the command line arguements are also below.

```
taskandyeshallperceive predict \
  --model-dir={modeldirpath}
  --input-csv={inputcsvpath}
  --n-classes=7 \
  --output-dir={outputdipath} \
  --target=0 \
  --n-samples=1\
  --noise-level=0.0
```

- `--model-dir`: the directory in which to save model checkpoints.
- `--input-csv`: path to CSV file with a row for each input with the first column containing the file path to a beta volume nifti file, the secind column containing a subject ID, and the third column containing the numerical code for the decoding target. The file must have a header, although the column names can be arbitrary.
- `--output-dir`: the directory in which to save prediction output files.
- `--n-classes`: the number of classes being decoded.
- `--target`: the numerical code for the decoding task for which you want to generate gradients. The default is "None", which will set the target to the correct target value for each example in the input CSV file.
- `--n-samples`: the number of samples to use when generating SmoothGrad gradients. The default is "1", which will disable SmoothGrad.
- `--noise-level`: the noise level (i.e. standard devation) of the noise used when generating SmoothGrad gradients. The default is 0.0, which will disable SmoothGrad.
