from typing import List

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tokenizer import get_chat_format

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('./saved_model')
tokenizer = AutoTokenizer.from_pretrained('./saved_model')

ASSISTANT_PROMPT = "<|im_start|>assistant\n"
EOS_TOKEN = "<|im_end|>"


def predict(element) -> List[str]:
    """
    This function predicts the response SQL query for a given user prompt.
    :param element: a dictionary containing instruction(userPrompt), input(DDL), response(SQl response)
    :return: a list of decoded strings
    """
    formatted = tokenizer.apply_chat_template(get_chat_format(element), tokenize=False,)
    formatted += ASSISTANT_PROMPT
    eos_token_id = tokenizer.get_vocab()[EOS_TOKEN]
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, eos_token_id=eos_token_id, max_new_tokens=1000)
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
    return response

