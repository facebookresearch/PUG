# PUG: Photorealistic and Semantically Controllable Synthetic Data for Representation Learning

<font size=9><div align='center'><b>PUG</b>: <b>P</b>hotorealistic <b>U</b>nreal <b>G</b>raphics</div></font>

<font size=9><div align='center' > <a href=https://pug.metademolab.com>**Website**</a> | <a href=https://arxiv.org/abs/2308.03977>**Research Paper**</a> | <a href=https://pug.metademolab.com/faq.html>**Datasheet**</a> </div></font>

https://github.com/facebookresearch/PUG/assets/5903040/463201e6-a831-44de-ad25-151663ec3761

This codebase contains:
- download links for the PUG-datasets
- dataloaders
- scripts that are needed to samples images from a running interactive environment made with the Unreal Engine.
- script to evaluate VLMs models with PUG: SPAR
- list of the assets used to create the PUG datasets (which are listed in each PUG folders)

## Downloading the PUG datasets
Here are the links to download the PUG datasets:
- [PUG: Animals (78GB)](https://dl.fbaipublicfiles.com/large_objects/pug/PUG_ANIMAL.tar.gz)
- [PUG: ImageNet (27GB)](https://dl.fbaipublicfiles.com/large_objects/pug/PUG_IMAGENET.tar.gz)
- [PUG: SPAR (16GB)](https://dl.fbaipublicfiles.com/large_objects/pug/PUG_SPAR.tar.gz)
- [PUG: AR4T (97GB)](https://dl.fbaipublicfiles.com/large_objects/pug/PUG_AR4T.tar.gz)

## Dataset loaders
Please look at each PUG subfolder to get information on how to load the datasets.

[PUG Animals](./PUG_Animals)

[PUG ImageNet](./PUG_ImageNet)

[PUG SPAR](./PUG_SPAR)

[PUG AR4T](./PUG_AR4T)

## How to create a PUG environment ?
The instruction are availables in the [torchmultiverse](./torchmultiverse) folder.

## LICENSE
The datasets are distributed under the CC-BY-NC, with the addenda that they should not be used to train Generative AI models, as found in the LICENSE file.

## Citing PUG
If you use the PUG datasets, please cite:
```
@misc{bordes2023pug,
      title={PUG: Photorealistic and Semantically Controllable Synthetic Data for Representation Learning}, 
      author={Florian Bordes and Shashank Shekhar and Mark Ibrahim and Diane Bouchacourt and Pascal Vincent and Ari S. Morcos},
      year={2023},
      eprint={2308.03977},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```