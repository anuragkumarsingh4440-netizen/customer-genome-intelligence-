import os
import json

base_path = os.getcwd()
notebooks_dir = os.path.join(base_path, "notebooks")

os.makedirs(notebooks_dir, exist_ok=True)

notebook_names = [
    "01_data_cleaning.ipynb",
    "02_clustering_analysis.ipynb",
    "03_model_building.ipynb"
]

empty_notebook = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

for name in notebook_names:
    file_path = os.path.join(notebooks_dir, name)
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(empty_notebook, f)

print("All 3 notebooks created successfully inside notebooks/ folder.")
