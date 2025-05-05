import json
import tkinter as tk

import numpy as np
import tensorflow as tf
from keras.models import load_model
from text_vectors import get_src_target
from transformer_layers import (
    PositionalEmbedding,
    TransformerDecoder,
    TransformerEncoder,
)

# Load model and vectorizers
transformer = load_model(
    "PATH_TO_YOUR_MODEL",
    compile=False,
    custom_objects={
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerEncoder": TransformerEncoder,
        "TransformerDecoder": TransformerDecoder,
    },
)

vectors = get_src_target()

source_vectorization = vectors.get("source")
target_vectorization = vectors.get("target")

fin_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(fin_vocab)), fin_vocab))
max_decoded_sentence_length = 20


def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    tokens = decoded_sentence.split()
    tokens = [t for t in tokens if t not in ("[start]", "[end]")]
    return " ".join(tokens)


root = tk.Tk()
root.geometry("400x400")
root.title("Translator 1.0")

# Variables
text_var = tk.StringVar()


# Functionalities
def translate():
    text = text_entry.get("1.0", tk.END).strip()
    traslation = decode_sequence(text)
    result_label.config(text=traslation)
    text_entry.delete("1.0", tk.END)


# Interaction
label = tk.Label(root, text="Type english text:", font=("JetBrains Mono", 12))
label.pack(pady=10)

text_entry = tk.Text(root, font=("JetBrains Mono", 10), height=5, width=40)
text_entry.pack()

button = tk.Button(root, text="Translate", command=translate)
button.pack(pady=10)

result_label = tk.Label(root, text="", font=("JetBrains Mono", 12))
result_label.pack(pady=10)

root.mainloop()
