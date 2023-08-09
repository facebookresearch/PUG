# PUG: SPAR

![PUG SPAR images](./PUG_SPAR_Github.png)

The PUG: SPAR (Scene, Position, Attribute, Relation) dataset is used for the evaluation of vision-language models. It demonstrates how synthetic data can be used to address current benchmarks limitations.

It includes:
- 43,560 pre-rendered images
- 10 backgrounds
- 32 animals
- 4 relations (left/right, bottom/top)
- 4 attributes (blue/red, grass/stone)

An example of dataloader can found in the file [PUG_SPAR.ipynb](./PUG_SPAR.ipynb).

## Running the VLMs evaluation on PUG: SPAR

In the file [run_eval_vlms_on_spar.py](./run_eval_vlms_on_spar.py) we present the code that is needed to reproduce the table in Figure 6 in the paper.
In this script, we leverage the factor of variations that are contained in the .csv labels file to generate captions. Then we perform caption retrieval (over all or in an hard negative settings) to evaluate the models. This script leverage [OpenCLIP](https://github.com/mlfoundations/open_clip) and the model zoo folder of [ARO](https://github.com/mertyg/vision-language-models-are-bows/tree/main/model_zoo). So before running the script, you should **copy the model zoo and misc folder from ARO into this folder** if you want to run the evaluation on BLIP, NegCLIP and X-VLM.

To eval CLIP (OpenAI) with a VIT-Base architecture, you can use the following:
```
python run_eval_vlms_on_spar.py PUG_SPAR_PATH --arch ViT-B-32 --model openai
```

### Credits

The list of the assets used to create PUG: ImageNet is available in the file [list_assets_pug_spar.txt](./list_assets_pug_spar.txt).
