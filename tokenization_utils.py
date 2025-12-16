from transformers import AutoTokenizer
muril_tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

def tokenize_and_match_word_ids_with_labels(examples):
    tokenized_inputs = muril_tokenizer(examples["मिश्रित_वाक्य"], truncation=True, is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids()

    current_labels = examples["चिन्ह"]
    new_labels = []

    # Loop through the batch
    for idx, each_labels_list in enumerate(current_labels):
        current_word_ids = tokenized_inputs.word_ids(idx)

        current_index = None
        new_labels_per_batch = []

        # Now loop through the word ids of the current sentence in the batch
        for each_word_id in current_word_ids:
            if each_word_id is None:                # -100 is the default value for padding/special tokens

                new_labels_per_batch.append(-100)
            elif each_word_id != current_index:
                # New word

                # each_word_id is the index of the word in the sentence
                current_index = each_word_id
                new_labels_per_batch.append(each_labels_list[each_word_id])
            else:
                # Same word different token (second or inner token just inside the same word)
                # For now let's just label it -100 so we only keep the label of the first token
                label_to_append = each_labels_list[each_word_id]
                label_to_append += 1 # change the label to its inner version
                new_labels_per_batch.append(label_to_append)
        
        new_labels.append(new_labels_per_batch)
    
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs