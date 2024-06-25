from typing import Tuple

import evaluate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import CHAT_ML_TEMPLATE

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.4"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, eos_token="<|im_end|>")
tokenizer.chat_template = CHAT_ML_TEMPLATE

bleu = evaluate.load("bleu")
acc = evaluate.load("accuracy")

# We’ll use the BLEU score and token-level accuracy to compare the model’s output tokens with the correct tokens
# (the inputs shifted by 1).


def preprocess_logits_for_metrics(logits: Tuple) -> torch.Tensor:
    """
    This function processes the logits output from a model to extract the predicted token IDs. It handles the case where
    logits might be a tuple (such as when the model outputs additional tensors like `past_key_values`), ensuring that
    only the first element (logits) is used.
    :param logits: The output logits from the model. If it's a tuple, the first element is assumed to be the logits
    :return: The predicted token IDs obtained by applying argmax over the last dimension of the logits tensor.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds: Tuple) -> dict:
    """
    This function calculates the BLEU score and accuracy for the predicted tokens against the true labels. It handles
    shifting the labels and predictions appropriately, masking ignored indices, and decoding token IDs to text for
    BLEU score computation.
    :param eval_preds: predicted token IDs from the model and ground truth token IDs
    :return: A dictionary containing the BLEU score and accuracy metrics.
    """
    preds, labels = eval_preds
    labels = labels[:, 1:]
    preds = preds[:, :-1]

    # -100 is a default value for ignore_index used by DataCollatorForCompletionOnlyLM
    mask = labels == -100
    # replace -100 with a value that the tokenizer can decode
    labels[mask] = tokenizer.pad_token_id
    preds[mask] = tokenizer.pad_token_id

    # bleu takes in text, so we have to translate from token ids to text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    # accuracy takes in lists of integers,
    # and we want to evaluate only the parts that are not -100,
    # hence the mask negation (~)
    accuracy = acc.compute(predictions=preds[~mask], references=labels[~mask])

    return {**bleu_score, **accuracy}
