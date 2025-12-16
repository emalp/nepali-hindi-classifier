from datasets import load_dataset as hf_load_dataset
import random

def hindi_cleaner(datasetdict):
    del datasetdict["validation"]
    datasetdict = datasetdict.remove_columns(["idx", "src"])
    datasetdict = datasetdict.rename_column("tgt", "हिंदी")
    datasetdict = datasetdict.shuffle(seed=42)
    datasetdict["train"] = datasetdict["train"].select(range(46432))
    return datasetdict

def generate_datasets():
    हिंदी = hf_load_dataset("rd124/samanantar_100K_hindi")
    final_hindi = hindi_cleaner(हिंदी)
    नेपाली_वाक्यानी = hf_load_dataset("text", data_files="नेपाली_वाक्यानी.txt")
    नेपाली_वाक्यानी = नेपाली_वाक्यानी.rename_column("text", "नेपाली")
    नेपाली_वाक्यानी["train"] = नेपाली_वाक्यानी["train"].add_column("हिंदी", final_hindi["train"]["हिंदी"])
    return नेपाली_वाक्यानी


def label_dataset(dataset):
    nepali_len = len(dataset["नेपाली"].split())
    hindi_len = len(dataset["हिंदी"].split())
    dataset["नेपाली_label"] = [0] * nepali_len
    dataset["हिंदी_label"] = [2] * hindi_len
    return dataset

def mix_nepali_hindi_dataset(hn_dataset):
    hn_dataset_mixed = list(zip(hn_dataset["नेपाली"].split() + hn_dataset["हिंदी"].split(), hn_dataset["नेपाली_label"] + hn_dataset["हिंदी_label"]))
    random.shuffle(hn_dataset_mixed)
    hindi_nepal_mixed_sentence, hindi_nepal_mixed_label = zip(*hn_dataset_mixed)
    #joined_sentence = " ".join(hindi_nepal_mixed_sentence)
    hn_dataset_mixed = {"मिश्रित_वाक्य": hindi_nepal_mixed_sentence, "चिन्ह": hindi_nepal_mixed_label}
    return hn_dataset_mixed

def generate_mixed_hindi_nepali_dataset():
    hindi_nepali_dataset = generate_datasets()
    hindi_nepali_dataset_labelled = hindi_nepali_dataset.map(label_dataset, batched=False)
    mixed_dataset = hindi_nepali_dataset_labelled.map(mix_nepali_hindi_dataset, batched=False)
    mixed_dataset_clean = mixed_dataset.remove_columns(["नेपाली", "हिंदी", "नेपाली_label", "हिंदी_label"])
    return mixed_dataset_clean