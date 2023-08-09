# PUG: ImageNet

![PUG_ImageNet images](./PUGImageNet_Github.png)

The PUG: ImageNet dataset provides a novel, useful benchmark for the fine-grained evaluation of the image classifiers’ robustness along several variation factors. It contains:

- 88,328 pre-rendered images using 724 assets representing 151 ImageNet classes
- 64 backgrounds
- 7 sizes
- 10 textures
- 18 camera orientations
- 18 character orientations
- 7 light intensities

The dataset comes with a csv annotation file which contain the following factor values:
world_name, character_name, character_label, character_rotation_yaw, character_rotation_roll,
character_rotation_pitch, character_scale, camera_roll, camera_pitch, camera_yaw, character_texture, 
scene_light

An example of dataloader can found in the file [PUG_ImageNet.ipynb](./PUG_ImageNet.ipynb).

### Credits

The list of the assets used to create PUG: ImageNet is available in the file [list_assets_pug_imagenet.txt](./list_assets_pug_imagenet.txt).
