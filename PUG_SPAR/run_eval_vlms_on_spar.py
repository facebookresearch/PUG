"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image, ImageFont, ImageDraw
import timm
from transformers import CLIPModel, CLIPProcessor
import open_clip
import pandas as pd
import numpy as np
import os
import torchmetrics
import argparse
import json
import csv
import itertools
from transformers import FlavaProcessor, FlavaForPreTraining, BertTokenizer, FlavaFeatureExtractor
import random

# Fix seed for reproducibility
torch.manual_seed(0)
random.seed(0)


# Wrapper for the Unreal Dataset
class UnrealDataset(torch.utils.data.Dataset):
    def __init__(self, df, labels, images_folder, transform = None, neg_labels = None):
        self.df = df
        self.labels = labels
        self.neg_labels = neg_labels
        self.images_folder = images_folder
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, index):
        filename = self.df['filename'][index]
        image = Image.open(os.path.join(self.images_folder, filename))
        if self.transform is not None:
            image = self.transform(image)
        if self.neg_labels is not None:
            return image, self.labels[index], self.neg_labels[index]
        else:
            return image, self.labels[index]            

# We define a serie of functions to process the images dependently of the model choosen
def process_blip_image(model, images):
    image_feat = model.model.visual_encoder(images)   
    image_embed = model.model.vision_proj(image_feat[:,0,:])            
    image_embed /= image_embed.norm(dim=-1, keepdim=True)
    return image_embed

def process_flava_image(model, images):
    inputs = model.feature_extractor(images=images, return_tensors="pt").to("cuda:0")
    image_embed = model.flava.get_image_features(**inputs)[:, 0, :]
    image_embed /= image_embed.norm(dim=-1, keepdim=True)
    return image_embed

def process_xvlm_image(model, images):
    image_feat = model.model.vision_encoder(images)   
    image_embed = model.model.vision_proj(image_feat[:,0,:]) 
    image_embed /= image_embed.norm(dim=-1, keepdim=True)           
    return image_embed

def process_negclip_image(model, images):
    image_embed = model.model.encode_image(images)
    image_embed /= image_embed.norm(dim=-1, keepdim=True)
    return image_embed

def process_clip_image(model, images):
    image_embed = model.encode_image(images)
    image_embed /= image_embed.norm(dim=-1, keepdim=True)
    return image_embed

# Main function to compute the cosine similarity between the images and captions
def process(model, tokenizer, dataloader, captions_full, hard_negative=False):
    # Create metrics
    if hard_negative:
        # We perform retrieval with hard negative, thus having only two captions by images
        m_acc_top1 = torchmetrics.Accuracy('multiclass', num_classes=2,compute_on_step=False).cuda()
        m_acc_top5 = torchmetrics.Accuracy('multiclass', num_classes=2,compute_on_step=False).cuda()
    else:
        # We perform retrieval over all captions availables.
        m_acc_top1 = torchmetrics.Accuracy('multiclass', num_classes=len(captions_full),compute_on_step=False).cuda()
        m_acc_top5 = torchmetrics.Accuracy('multiclass', num_classes=len(captions_full),compute_on_step=False, top_k=5).cuda()

    # Compute text features (which might differ accross models)
    if model.name == "clip":
        list_features = []
        for i in range(int(len(captions_full)/128)+1):
            with torch.no_grad(), torch.cuda.amp.autocast():
                text = tokenizer(captions_full[i*128:i*128+128])
                text_features = model.encode_text(text.to("cuda:0"))#.cuda().half()
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.cpu()
                list_features.append(text_features)
        text_features = torch.cat(list_features, dim=0).to("cuda:0")
        process_image = process_clip_image
    elif model.name == "negclip":
        text_features = model.get_text_embeddings(captions_full, normalize=True)
        process_image = process_negclip_image
    elif model.name == "x-vlm":
        text_features, text_ids, text_atts = model.get_text_embeddings(captions_full)
        process_image = process_xvlm_image
    elif model.name == "flava":
        list_features = []
        for i in range(int(len(captions_full)/128)+1):
            with torch.no_grad(), torch.cuda.amp.autocast():
                if len(captions_full[i*128:i*128+128]) > 0:
                    text = tokenizer(text=captions_full[i*128:i*128+128], return_tensors="pt", padding="max_length", max_length=77)
                    text_features = model.flava.get_text_features(**text.to("cuda:0"))[:, 0, :]
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_features = text_features.cpu()
                    list_features.append(text_features)
        text_features = torch.cat(list_features, dim=0)
        process_image = process_flava_image
    elif model.name == "blip":
        text_features, text_ids, text_atts = model.get_text_embeddings(captions_full)
        process_image = process_blip_image
    # Move text feature to gpu
    text_features = text_features.to("cuda:0")

    # Run the loop through all the images/text
    list_incorrect = []
    list_bad_pred = []
    for k, data in enumerate(dataloader):
        if len(data) == 2:
            images, index = data
            neg_index = None
        else:
            images, index, neg_index = data
        label = index.to("cuda:0")
        if type(images) is not tuple:
            images = images.to("cuda:0")
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Compute image representation
            image_embed = process_image(model, images)
            # Compute cosine similarity between the image and text embeddings     
            if neg_index is not None:
                # Compute only with the correct and hard negative captions
                if random.random() >= 0.5: # Important trick to ensure that a collapse encoder to not always give the correct answer
                    label = torch.tensor(1).to("cuda:0").unsqueeze(0)
                    output = (100.0 * image_embed @ text_features[[neg_index.item(), index.item()]].t()).softmax(dim=-1)
                else:
                    label = torch.tensor(0).to("cuda:0").unsqueeze(0)
                    output = (100.0 * image_embed @ text_features[[index.item(), neg_index.item()]].t()).softmax(dim=-1)            
            else:
                # Otherwise compute using every captions
                output = (100.0 * image_embed @ text_features.t()).softmax(dim=-1)
            # Compute the argmax
            pred = torch.argmax(output, dim=1)
            # Compute metrics
            if pred != label:
                # We save the list of the miscaptioned images
                list_incorrect.append(k)
                list_bad_pred.append(pred.item())
            m_acc_top1(output, label)
            m_acc_top5(output, label)
    final_acc1 = m_acc_top1.compute().item() * 100
    final_acc5 = m_acc_top5.compute().item() * 100
    return final_acc1, final_acc5, list_incorrect, list_bad_pred

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='evaluatings VLMs with PUG: SPAR')
    parser.add_argument('dataset_path')
    parser.add_argument('-a','--arch', help='Atchitecture of the model', required=True)
    parser.add_argument('-m','--model', help='Name of the model', required=True)
    parser.add_argument('-e','--env', help='Name of the environment', default="All", required=False)
    args = parser.parse_args()

    # Set data transformations
    tr_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    # Crop small
    transform_zoom = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        tr_normalize,
    ])

    # Load csv file
    main_df = pd.read_csv(args.dataset_path+"labels.csv")

    # Create model
    collate_fn = None
    if args.model == "blip":
        # You should import model_zoo from
        # https://github.com/mertyg/vision-language-models-are-bows/tree/main/model_zoo
        from model_zoo import get_model
        model, transform_zoom = get_model(model_name="blip-flickr-base", device="cuda", root_dir="")
        tokenizer = None
        model.name = "blip"
    elif args.model == "flava":
        # You should import misc from
        # https://github.com/mertyg/vision-language-models-are-bows/blob/main/misc/__init__.py
        from misc import _default_collate
        model = FlavaForPreTraining.from_pretrained("facebook/flava-full").eval().to("cuda:0")
        model.feature_extractor = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")
        tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")
        processor = FlavaProcessor.from_pretrained("facebook/flava-full")
        collate_fn=_default_collate
        transform_zoom=None
        model.name = "flava"
    elif args.model == "x-vlm":
        from model_zoo import get_model
        model, transform_zoom = get_model(model_name="xvlm-flickr", device="cuda", root_dir="")
        tokenizer = None
        model.name = "x-vlm"
    elif args.model == "negclip":
        from model_zoo import get_model
        model, transform_zoom = get_model(model_name="NegCLIP", device="cuda", root_dir="")
        tokenizer = None
        model.name = "negclip"
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(args.arch, pretrained=args.model, precision="fp16", device="cuda:0")
        tokenizer = open_clip.get_tokenizer(args.arch)
        model.name = "clip"


    #################################################################
    ##################### SCENE RECOGNITION #####################
    #################################################################

    ### Prepare background captions
    dict_result = {}
    subset_background_only = main_df[main_df['character_name'] == 'blank'].reset_index(drop=True)
    subset_background_only = subset_background_only[subset_background_only['character2_name'] == 'blank'].reset_index(drop=True)
    subset_background_only = subset_background_only[:10]
    list_worlds = subset_background_only['world_name'].unique().tolist()
    # Defines captions
    captions_worlds = []
    for c in list_worlds:
        captions_worlds.append(f"This is a photo in a {c} environment")
    # Define labels
    labels = []
    for idx in range(len(subset_background_only)):
        labels.append(captions_worlds.index(f"This is a photo in a {subset_background_only['world_name'][idx]} environment"))
    # Prepare dataset
    dataset = UnrealDataset(subset_background_only, labels, images_folder=args.dataset_path, transform=transform_zoom)
    dataloader_backgrounds = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, collate_fn=collate_fn)
    # Run the model
    accs = process(model, tokenizer, dataloader_backgrounds, captions_worlds)
    dict_result[f"scene_backgrounds_top1"] = accs[0]
    print("Accuracy scene recognition: ", accs)

    ### Prepare for single object detection
    list_worlds = main_df['world_name'].unique().tolist()
    # By default we compute accuracy by background then over all background
    if args.env == "All":
        indexes_world = range(len(list_worlds) + 2)
    # we can also run the script for a specific environment
    elif args.env in list_worlds:
        indexes_world = list_worlds.index(args.env)
    # Otherwise we compute only across all
    else:
        indexes_world = 11
    # Loop through environment
    for idx in indexes_world:
        world = None
        add_background_name = False
        if idx < len(list_worlds):
            world = list_worlds[idx]
        elif idx == len(list_worlds)+1:
            add_background_name = True

        #################################################################
        ##################### SINGLE ANIMAL SETTING #####################
        #################################################################

        # We take a data subset for single object detection (We made sure to have only a single animal on the images)
        subset_char1_only = main_df[(main_df['character_name'] == 'blank') & (main_df['character2_name'] != 'blank')]
        subset_char2_only = main_df[(main_df['character2_name'] == 'blank') & (main_df['character_name'] != 'blank')]
        subset_char_only = pd.concat([subset_char1_only, subset_char2_only]).reset_index(drop=True)
        if world is not None:
            subset_char_only = subset_char_only[subset_char_only['world_name'] == world].reset_index(drop=True)
        # subset_char_only = subset_char_only[(subset_char_only['character_texture'] == texture)].reset_index(drop=True)

        # We take a different subset by texture
        for texture in ['Default', 'Blue', 'Grass']:
            subset_char_only_by_texture = subset_char_only[(subset_char_only['character_texture'] == texture)].reset_index(drop=True)
            list_char = subset_char_only_by_texture['character_name'].unique().tolist()
            # Defines captions
            captions_char_only = []
            for c in list_char:
                if add_background_name:
                    for w in list_worlds:
                        captions_char_only.append(f"This is a photo of a {c} in a {w} environment")
                else:
                    captions_char_only.append(f"This is a photo of a {c}")
            # Define labels
            labels = []
            for idx in range(len(subset_char_only_by_texture)):
                if subset_char_only_by_texture['character_name'][idx] != 'blank':
                    if add_background_name:
                        labels.append(captions_char_only.index(f"This is a photo of a {subset_char_only_by_texture['character_name'][idx]} in a {subset_char_only_by_texture['world_name'][idx]} environment"))
                    else:
                        labels.append(captions_char_only.index(f"This is a photo of a {subset_char_only_by_texture['character_name'][idx]}"))
                elif subset_char_only_by_texture['character2_name'][idx] != 'blank':
                    if add_background_name:
                        labels.append(captions_char_only.index(f"This is a photo of a {subset_char_only_by_texture['character2_name'][idx]} in a {subset_char_only_by_texture['world_name'][idx]} environment"))
                    else:
                        labels.append(captions_char_only.index(f"This is a photo of a {subset_char_only_by_texture['character2_name'][idx]}"))
            # Prepare dataset
            dataset = UnrealDataset(subset_char_only_by_texture, labels, images_folder=args.dataset_path, transform=transform_zoom)
            dataloader_backgrounds = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=collate_fn)
            # Run the model
            accs = process(model, tokenizer, dataloader_backgrounds, captions_char_only)
            dict_result[f"{world}_captionback_{str(add_background_name)}_one_objects_{texture}_len"] = len(captions_char_only)
            dict_result[f"{world}_captionback_{str(add_background_name)}_one_objects_{texture}_top1"] = accs[0]
            dict_result[f"{world}_captionback_{str(add_background_name)}_one_objects_{texture}_top5"] = accs[1]
            dict_result[f"{world}_captionback_{str(add_background_name)}_one_objects_{texture}_indexes_bad"] = accs[2]
            dict_result[f"{world}_captionback_{str(add_background_name)}_one_objects_{texture}_pred_bad"] = accs[3]
            print("Accuracy single animal detection: ", world, texture, accs)

        # Prepare single object detections with relations
        for relation in [("left", "right"), ("bottom", "top")]:
            r1, r2 = relation
            subset_char_only_relation = subset_char_only[(subset_char_only['character_texture'] == 'Default')].reset_index(drop=True)
            subset_char_only_relation = subset_char_only_relation[(subset_char_only_relation['character1_pos'] == r1)].reset_index(drop=True)
            list_char = subset_char_only_relation['character_name'].unique().tolist()[1:]
            # Define captions
            captions_char_only = []
            for c in list_char:
                if add_background_name:
                    for w in list_worlds:
                        for r in relation:
                            captions_char_only.append(f"This is a photo of a {c} on the {r} of the picture in a {w} environment")
                else:
                    for r in relation:
                        captions_char_only.append(f"This is a photo of a {c} on the {r} of the picture")
            # Define labels
            labels = []
            neg_labels = []
            for idx in range(len(subset_char_only_relation)):
                if subset_char_only_relation['character_name'][idx] != 'blank':
                    if add_background_name:
                        labels.append(captions_char_only.index(f"This is a photo of a {subset_char_only_relation['character_name'][idx]} on the {r1} of the picture in a {subset_char_only_relation['world_name'][idx]} environment"))
                        neg_labels.append(captions_char_only.index(f"This is a photo of a {subset_char_only_relation['character_name'][idx]} on the {r2} of the picture in a {subset_char_only_relation['world_name'][idx]} environment"))
                    else:
                        labels.append(captions_char_only.index(f"This is a photo of a {subset_char_only_relation['character_name'][idx]} on the {r1} of the picture"))
                        neg_labels.append(captions_char_only.index(f"This is a photo of a {subset_char_only_relation['character_name'][idx]} on the {r2} of the picture"))
                elif subset_char_only_relation['character2_name'][idx] != 'blank':
                    if add_background_name:
                        labels.append(captions_char_only.index(f"This is a photo of a {subset_char_only_relation['character2_name'][idx]} on the {r2} of the picture in a {subset_char_only_relation['world_name'][idx]} environment"))
                        neg_labels.append(captions_char_only.index(f"This is a photo of a {subset_char_only_relation['character2_name'][idx]} on the {r1} of the picture in a {subset_char_only_relation['world_name'][idx]} environment"))
                    else:
                        labels.append(captions_char_only.index(f"This is a photo of a {subset_char_only_relation['character2_name'][idx]} on the {r2} of the picture"))
                        neg_labels.append(captions_char_only.index(f"This is a photo of a {subset_char_only_relation['character2_name'][idx]} on the {r1} of the picture"))
            # Prepare dataset
            dataset = UnrealDataset(subset_char_only_relation, labels, images_folder=args.dataset_path, transform=transform_zoom)
            dataloader_backgrounds = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=collate_fn)
            # Run the model
            accs = process(model, tokenizer, dataloader_backgrounds, captions_char_only)
            dict_result[f"{world}_captionback_{str(add_background_name)}_one_objects_{texture}_relations_{r1}_len"] = len(captions_char_only)
            dict_result[f"{world}_captionback_{str(add_background_name)}_one_objects_{texture}_relations_{r1}_top1"] = accs[0]
            dict_result[f"{world}_captionback_{str(add_background_name)}_one_objects_{texture}_relations_{r1}_top5"] = accs[1]
            dict_result[f"{world}_captionback_{str(add_background_name)}_one_objects_{texture}_relations_{r1}_indexes_bad"] = accs[2]
            dict_result[f"{world}_captionback_{str(add_background_name)}_one_objects_{texture}_relations_{r1}_pred_bad"] = accs[3]
            print("Accuracy single animal position: ", relation, world, texture, accs)
            # Prepare dataset
            dataset = UnrealDataset(subset_char_only_relation, labels, images_folder=args.dataset_path, transform=transform_zoom, neg_labels=neg_labels)
            dataloader_backgrounds = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=collate_fn)
            # Run the model
            accs = process(model, tokenizer, dataloader_backgrounds, captions_char_only, hard_negative=True)
            dict_result[f"{world}_captionback_{str(add_background_name)}_one_objects_{texture}_relations_{r1}_hardneg_top1"] = accs[0]
            print("Accuracy single animal position hard-negative: ", relation, world, texture, accs)

        #################################################################
        ##################### DOUBLE ANIMAL SETTING #####################
        #################################################################

        # We take a data subset for double object detection (We made sure to have two animals on the images)
        subset_char_double = main_df[(main_df['character_name'] != 'blank') & (main_df['character2_name'] != 'blank')].reset_index(drop=True)
        if world is not None:
            subset_char_double = subset_char_double[subset_char_double['world_name'] == world].reset_index(drop=True)

        # Prepare double object detections
        for texture in ['Default', 'Blue', 'Grass']:
            list_tuples = []
            subset_char_double_by_texture = subset_char_double[(subset_char_double['character_texture'] == texture)].reset_index(drop=True)
            list_char1 = subset_char_double_by_texture['character_name'].tolist()
            list_char2 = subset_char_double_by_texture['character2_name'].tolist()
            for index in range(len(subset_char_double_by_texture)):
                list_tuples.append(sorted([list_char1[index], list_char2[index]]))
            list_tuples.sort()
            list_objects = list(k for k,_ in itertools.groupby(list_tuples))
            # Define captions
            captions_two_objects = []
            for elem in list_objects:
                if add_background_name:
                    for w in list_worlds:
                        captions_two_objects.append(f"This is a photo of a {elem[0]} and a {elem[1]} in a {w} environment") 
                else:
                    captions_two_objects.append(f"This is a photo of a {elem[0]} and a {elem[1]}")
            # Define labels
            labels = []
            for idx in range(len(subset_char_double_by_texture)):
                actor1, actor2 = sorted([subset_char_double_by_texture['character_name'][idx], subset_char_double_by_texture['character2_name'][idx]])
                if add_background_name:
                    labels.append(captions_two_objects.index(f"This is a photo of a {actor1} and a {actor2} in a {subset_char_double_by_texture['world_name'][idx]} environment"))
                else:
                    labels.append(captions_two_objects.index(f"This is a photo of a {actor1} and a {actor2}"))
            # Prepare dataset
            dataset = UnrealDataset(subset_char_double_by_texture, labels, images_folder=args.dataset_path, transform=transform_zoom)
            dataloader_backgrounds = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=collate_fn)
            # Run the model
            accs = process(model, tokenizer, dataloader_backgrounds, captions_two_objects)
            dict_result[f"{world}_captionback_{str(add_background_name)}_two_objects_{texture}_len"] = len(captions_two_objects)
            dict_result[f"{world}_captionback_{str(add_background_name)}_two_objects_{texture}_top1"] = accs[0]
            dict_result[f"{world}_captionback_{str(add_background_name)}_two_objects_{texture}_top5"] = accs[1]
            dict_result[f"{world}_captionback_{str(add_background_name)}_two_objects_{texture}_indexes_bad"] = accs[2]
            dict_result[f"{world}_captionback_{str(add_background_name)}_two_objects_{texture}_pred_bad"] = accs[3]
            print("Accuracy double animals detection: ", world, texture, accs)
        
        # Prepare double object detections with relations left/right
        subset_char_double_by_texture = subset_char_double[(subset_char_double['character_texture'] == 'Default')].reset_index(drop=True)
        for pos in [("left", "right"), ("bottom", "top")]:
            list_tuples = []
            subset_char_double_relations = subset_char_double_by_texture[(subset_char_double_by_texture['character1_pos'] == pos[0])].reset_index(drop=True)
            subset_char_double_relations = subset_char_double_relations.loc[~(subset_char_double_relations['character_name'] == subset_char_double_relations['character2_name'])].reset_index(drop=True)
            list_char1 = subset_char_double_relations['character_name'].tolist()
            list_char2 = subset_char_double_relations['character2_name'].tolist()
            # Define captions
            captions_two_objects = []
            for idx in range(len(subset_char_double_relations)):
                if add_background_name:
                    for w in list_worlds:
                        captions_two_objects.append(f"This is a photo of a {subset_char_double_relations['character_name'][idx]} on the {pos[0]} and a {subset_char_double_relations['character2_name'][idx]} on the {pos[1]} of the picture in a {w} environment")
                else:
                    captions_two_objects.append(f"This is a photo of a {subset_char_double_relations['character_name'][idx]} on the {pos[0]} and a {subset_char_double_relations['character2_name'][idx]} on the {pos[1]} of the picture")
            captions_two_objects = list(set(captions_two_objects))
            # Define labels
            labels = []
            neg_labels = []
            for idx in range(len(subset_char_double_relations)):
                actor1, actor2 = subset_char_double_relations['character_name'][idx], subset_char_double_relations['character2_name'][idx]
                if add_background_name:
                    labels.append(captions_two_objects.index(f"This is a photo of a {actor1} on the {pos[0]} and a {actor2} on the {pos[1]} of the picture in a {subset_char_double_relations['world_name'][idx]} environment"))
                    neg_labels.append(captions_two_objects.index(f"This is a photo of a {actor2} on the {pos[0]} and a {actor1} on the {pos[1]} of the picture in a {subset_char_double_relations['world_name'][idx]} environment"))
                else:
                    labels.append(captions_two_objects.index(f"This is a photo of a {actor1} on the {pos[0]} and a {actor2} on the {pos[1]} of the picture"))
                    neg_labels.append(captions_two_objects.index(f"This is a photo of a {actor2} on the {pos[0]} and a {actor1} on the {pos[1]} of the picture"))
            # Prepare dataset
            dataset = UnrealDataset(subset_char_double_relations, labels, images_folder=args.dataset_path, transform=transform_zoom)
            dataloader_backgrounds = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=collate_fn)
            # Run the model
            accs = process(model, tokenizer, dataloader_backgrounds, captions_two_objects)
            print("Accuracy double animals position: ", pos, world, accs)
            dict_result[f"{world}_captionback_{str(add_background_name)}_relations_{pos[0]}_{pos[1]}_default_len"] = len(captions_two_objects)
            dict_result[f"{world}_captionback_{str(add_background_name)}_relations_{pos[0]}_{pos[1]}_default_top1"] = accs[0]
            dict_result[f"{world}_captionback_{str(add_background_name)}_relations_{pos[0]}_{pos[1]}_default_top5"] = accs[1]
            dict_result[f"{world}_captionback_{str(add_background_name)}_relations_{pos[0]}_{pos[1]}_default_indexes_bad"] = accs[2]
            dict_result[f"{world}_captionback_{str(add_background_name)}_relations_{pos[0]}_{pos[1]}_default_pred_bad"] = accs[3]
            # Prepare dataset
            dataset = UnrealDataset(subset_char_double_relations, labels, images_folder=args.dataset_path, transform=transform_zoom, neg_labels=neg_labels)
            dataloader_backgrounds = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=collate_fn)
            # Run the model
            accs = process(model, tokenizer, dataloader_backgrounds, captions_two_objects, hard_negative=True)
            print("Accuracy double animals position hard-negative: ", pos, world, accs)
            dict_result[f"{world}_captionback_{str(add_background_name)}_relations_{pos[0]}_{pos[1]}_hardneg_top1"] = accs[0]

        # Prepare double object detections with attributes
        for texture in ['Blue', 'Grass']:
            list_tuples = []
            subset_char_double_attribute = subset_char_double[(subset_char_double['character_texture'] == texture)]
            subset_char_double_attribute = subset_char_double_attribute.loc[~(subset_char_double_attribute['character_name'] == subset_char_double_attribute['character2_name'])].reset_index(drop=True)
            list_char1 = subset_char_double_attribute['character_name'].tolist()
            list_char2 = subset_char_double_attribute['character2_name'].tolist()
            # Define captions
            captions_two_objects = []
            for idx in range(len(subset_char_double_attribute)):
                if add_background_name:
                    for w in list_worlds:
                        captions_two_objects.append(f"This is a photo of a {subset_char_double_attribute['character_name'][idx]} textured with {subset_char_double_attribute['character_texture'][idx]} and a {subset_char_double_attribute['character2_name'][idx]} textured with {subset_char_double_attribute['character2_texture'][idx]} in a {w} environment")
                else:
                    captions_two_objects.append(f"This is a photo of a {subset_char_double_attribute['character_name'][idx]} textured with {subset_char_double_attribute['character_texture'][idx]} and a {subset_char_double_attribute['character2_name'][idx]} textured with {subset_char_double_attribute['character2_texture'][idx]}")
            captions_two_objects = list(set(captions_two_objects))
            # Define labels
            labels = []
            neg_labels = []
            for idx in range(len(subset_char_double_attribute)):
                if add_background_name:
                    labels.append(captions_two_objects.index(f"This is a photo of a {subset_char_double_attribute['character_name'][idx]} textured with {subset_char_double_attribute['character_texture'][idx]} and a {subset_char_double_attribute['character2_name'][idx]} textured with {subset_char_double_attribute['character2_texture'][idx]} in a {subset_char_double_attribute['world_name'][idx]} environment"))
                    neg_labels.append(captions_two_objects.index(f"This is a photo of a {subset_char_double_attribute['character2_name'][idx]} textured with {subset_char_double_attribute['character_texture'][idx]} and a {subset_char_double_attribute['character_name'][idx]} textured with {subset_char_double_attribute['character2_texture'][idx]} in a {subset_char_double_attribute['world_name'][idx]} environment"))
                else:
                    labels.append(captions_two_objects.index(f"This is a photo of a {subset_char_double_attribute['character_name'][idx]} textured with {subset_char_double_attribute['character_texture'][idx]} and a {subset_char_double_attribute['character2_name'][idx]} textured with {subset_char_double_attribute['character2_texture'][idx]}"))
                    neg_labels.append(captions_two_objects.index(f"This is a photo of a {subset_char_double_attribute['character2_name'][idx]} textured with {subset_char_double_attribute['character_texture'][idx]} and a {subset_char_double_attribute['character_name'][idx]} textured with {subset_char_double_attribute['character2_texture'][idx]}"))
            # Prepare dataset
            dataset = UnrealDataset(subset_char_double_attribute, labels, images_folder=args.dataset_path, transform=transform_zoom)
            dataloader_backgrounds = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=collate_fn)
            # Run the model
            accs = process(model, tokenizer, dataloader_backgrounds, captions_two_objects)
            dict_result[f"{world}_captionback_{str(add_background_name)}_attributes_{texture}_len"] = len(captions_two_objects)
            dict_result[f"{world}_captionback_{str(add_background_name)}_attributes_{texture}_top1"] = accs[0]
            dict_result[f"{world}_captionback_{str(add_background_name)}_attributes_{texture}_top5"] = accs[1]
            dict_result[f"{world}_captionback_{str(add_background_name)}_attributes_{texture}_indexes_bad"] = accs[2]
            dict_result[f"{world}_captionback_{str(add_background_name)}_attributes_{texture}_pred_bad"] = accs[3]
            print("Accuracy double animals attributes: ", world, texture, accs)
            # Prepare dataset
            dataset = UnrealDataset(subset_char_double_attribute, labels, images_folder=args.dataset_path, transform=transform_zoom, neg_labels=neg_labels)
            dataloader_backgrounds = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=collate_fn)
            # Run the model
            accs = process(model, tokenizer, dataloader_backgrounds, captions_two_objects, hard_negative=True)
            dict_result[f"{world}_captionback_{str(add_background_name)}_attributes_{texture}_hardneg_top1"] = accs[0]
            print("Accuracy double animals attributes hard-negative: ", pos, world, accs)
    
    # We save the file containing the values
    name_file = 'log_all_'+args.arch+'_'+ args.model
    print(dict_result)
    with open('logs/' + name_file, 'w+') as fd:
        fd.write(json.dumps(dict_result) + '\n')
        fd.flush()
