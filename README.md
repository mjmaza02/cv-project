# 6.S058 Project: Semantic Segmentation Model Prunning
Marcelo Maza, Darren Chen

## Installation
To get this project up and running, there are some steps that need to be taken. Run the following commands.

```
chmod +x download_ADE20K.sh && ./download_ADE20K.sh
chmod +x download_model.sh && ./download_model.sh
```

This downloads the `ADE20K` image set, as well as the `ResNet50dilated_PPM` model.

To use the pruner, make sure to add the following to any notebook:

```
!pip install ./acosp/
```