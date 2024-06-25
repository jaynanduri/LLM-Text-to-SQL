from typing import List

import datasets
from utils import create_dataset

from model import tokenizer


def get_chat_format(element: datasets.DatasetDict) -> List:
    """
    To make the model understand the dataset, we need to apply chat Template to every sample
    :param element: a dictionary containing instruction(userPrompt), input(DDL), response(SQl response)
    :return: updated user input with system prompt and DDL
    """
    system_prompt = (
        "You are a helpful programmer assistant that excels at SQL. "
        "When prompted with a task and a definition of an SQL table, you "
        "respond with a SQL query to retrieve information from the table. "
        "Don't explain your reasoning, only provide the SQL query."
    )
    user_prompt = "Task: {instruction}\nSQL table: {input}\nSQL query: "

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format_map(element)},
        {"role": "assistant", "content": element["response"]},
    ]


def tokenize(element: datasets.DatasetDict) -> dict:
    """
    Convert the samples to Tokens and encode them.
    :param element: a dictionary containing instruction(userPrompt), input(DDL), response(SQl response)
    :return: add new keys input_ids and attention mask to the original data dictionary
    """
    formatted = tokenizer.apply_chat_template(get_chat_format(element), tokenize=False)
    outputs = tokenizer(formatted)
    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"],
    }


dataset = create_dataset("hard")
for k in dataset.keys():
    dataset[k] = dataset[k].map(tokenize)
