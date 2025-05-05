import random
import re
import string

import tensorflow as tf
from keras import layers


def custom_standardization(input_string):
    strip_chars = string.punctuation
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")

    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")


def get_src_target():
    text_file = "C:\\Users\\miroh\\Documents\\koulu\\Git\\Neuralnetwork\\data\\fin.txt"

    with open(text_file, encoding="utf-8") as f:
        lines = f.read().split("\n")[:-1]

    text_pairs = []

    for line in lines:
        english, finnish, rest = line.split("\t")
        finnish = "[start] " + finnish + " [end]"
        text_pairs.append((english, finnish))

    random.shuffle(text_pairs)

    num_val_samples = int(0.15 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples

    train_pairs = text_pairs[:num_train_samples]
    val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples :]

    vocab_size = 50000
    seq_len = 40

    source_vectorization = layers.TextVectorization(
        max_tokens=vocab_size, output_mode="int", output_sequence_length=seq_len
    )

    target_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=seq_len + 1,
        standardize=custom_standardization,
    )

    train_eng_texts = [pair[0] for pair in train_pairs]
    train_fin_texts = [pair[1] for pair in train_pairs]

    source_vectorization.adapt(train_eng_texts)
    target_vectorization.adapt(train_fin_texts)

    return {"source": source_vectorization, "target": target_vectorization}
