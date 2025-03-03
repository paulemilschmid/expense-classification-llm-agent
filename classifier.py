#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classifier for excel expense categories using Google Gemini model.
"""

import os
import re
import yaml
import pandas as pd
from tqdm import tqdm
import time

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# -----

def load_prefixes(file_path="prefixes.yaml"):
    """
    Load prefixes from a YAML file.
    """
    with open(file_path, "r") as pfile:
        prefixes_config = yaml.safe_load(pfile)
    return prefixes_config["prefixes"]


def load_category_keywords(file_path="categories.yaml"):
    """
    Load categories and associated keywords from a YAML file.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    categories_keywords = config["categories"]
    
    keywords_string = ""
    for cat, keywords in categories_keywords.items():
        keywords_string += f"{cat}: {', '.join(keywords)}\n"
    
    return categories_keywords, keywords_string



def preprocess_text(text: str, prefixes) -> str:
    """
    Remove irrelevant prefixes, addresses, or other noise from text.
    """
    cleaned = text.lower()
    
    for pattern in prefixes:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    cleaned = re.sub(r'\bch-\d{4}\b', '', cleaned)  # CH-xxxx
    cleaned = re.sub(r'\b\d{4,5}\b', '', cleaned)    # 4- or 5-digit zip codes
    cleaned = re.sub(r'\bnan\b', '', cleaned)        # remove literal 'nan'
    
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned



def get_class(desc, keywords_string):
    """
    Get expense category for a given description from google gemini model. 
    """

    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ["GOOGLE_GEMINI_API_KEY"],
    )

    content_system = (
        "Ich möchte meine Ausgaben in bestimmte Kategorien einordnen. "
        "Du erhältst eine Liste von Kategorien und deren zugehörigen Schlüsselwörtern. "
        "Ordne bitte die jeweilige Ausgabe genau einer der aufgelisteten Kategorien zu und antworte ausschließlich mit dem exakten Namen der Kategorie. "
        "Kategorien und Schlüsselwörter:\n" + keywords_string
    )


    completion = client.chat.completions.create(
        model="models/gemini-2.0-flash-lite",
        messages=[
            {"role": "system", "content": content_system},
            {"role": "user", "content": "Expense description: " + str(desc)}
        ]
    )
    
    if not completion or not hasattr(completion, 'choices') or not completion.choices:
        print("No valid response from API:", completion)
        return None

    return completion.choices[0].message.content.strip()




def main():

    in_path = input("Enter the path to the excel file: ")
    out_path = input("Enter the path to the output file: ")

    prefixes = load_prefixes()
    categories_keywords, keywords_string = load_category_keywords()

    df = pd.read_excel(in_path)

    df["combined"] = df["Booking text"].astype(str) + " " + df["Payment purpose"].astype(str)
    df["cleaned_text"] = df["combined"].apply(lambda x: preprocess_text(x, prefixes=prefixes))

    new_labels = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]): 
        label = get_class(row["cleaned_text"], keywords_string)
        new_labels.append(label)
        time.sleep(2)
    df["category"] = new_labels

    df.to_excel(out_path, index=False)

    print('finished!')




if __name__ == "__main__":  
    main()