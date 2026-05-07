# -*- coding: utf-8 -*-
"""Tabular and image-similarity helpers used during dataset prep / diagnostics."""
import pandas as pd
from PIL import Image
from numpy import average, dot, linalg

__all__ = [
    'read_text',
    'read_text_LV',
    'get_thum',
    'img_similarity_vectors_via_numpy',
]


def read_text(filename):
    # Padding is handled downstream by the HF tokenizer ([PAD] + attention_mask=0).
    # The legacy ' EOF XXX' word-padding existed only because bert-embedding had
    # no PAD token; with subword tokenizers it would burn real seq_len slots.
    df = pd.read_excel(filename)
    text = {}
    for i in df.index.values:
        text[df.Image[i]] = df.Description[i]
    return text


def read_text_LV(filename):
    df = pd.read_excel(filename)
    text = {}
    for i in df.index.values:  # Gets the index of the row number and traverses it
        count = len(df.Description[i].split())
        if count < 30:
            df.Description[i] = df.Description[i] + ' EOF XXX' * (20 - count)  # LV_loss: 24
        text[df.Image[i]] = df.Description[i]
    return text  # return dict (key: values)


# Unification images processing
def get_thum(image, size=(224, 224), greyscale=False):
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image


# Calculate the cosine distance between pictures
def img_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_turple in image.getdata():
            vector.append(average(pixel_turple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res
