import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


from .generate_qa import *

# TODO: implement validate_generateqa
def validate_generate_qa(json_path:str):
    """
    Validate the generate_qa.py script by generating questions and answers for a sample image.

    Args:
        split (str): The split of the dataset to use ('train', 'valid_grader', 'train_demo').
        data_dir (str): The directory containing the dataset.
    """
    # Load the dataset
    with open(json_path, 'r') as f:
        data = json.load(f)

    for qa in data:
        image_path = qa['image_path']
        question = qa['question']
        answer = qa['answer']

        tokens = image_path.split("valid/")[1]
        tokens = tokens.split("_")
        info_prefix = tokens[0]
        view_index = tokens[1]
        info_path = f"valid/{info_prefix}_info.json"
        check_qa_pairs(info_path, view_index)



    # Print the results
    print(f"Image Path: {image_path}")
    print(f"Question: {question}")
    print(f"Answer: {answer}")