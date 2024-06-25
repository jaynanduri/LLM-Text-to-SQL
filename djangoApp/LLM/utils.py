import datasets
import pandas as pd

CHAT_ML_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'user' %}
{{'<|im_start|>user\n' + message['content'].strip() + '<|im_end|>' }}
{% elif message['role'] == 'system' %}
{{'<|im_start|>system\n' + message['content'].strip() + '<|im_end|>' }}
{% elif message['role'] == 'assistant' %}
{{'<|im_start|>assistant\n'  + message['content'] + '<|im_end|>' }}
{% endif %}
{% endfor %}
"""


def add_length_column(dataset: datasets.Dataset) -> pd.DataFrame:
    """
    This function converts dataset object into a dataframe.
    :param dataset: datasets.DatasetDict
    :return:
    """
    df = dataset.to_pandas()
    df["total_length"] = 0
    for column_name in ["instruction", "input", "response"]:
        num_words = df[column_name].astype(str).str.split().apply(len)
        df["total_length"] += num_words

    return df


def filter_by_total_length(df: pd.DataFrame, difficulty: str, numSamples: int) -> pd.DataFrame:
    """
    This function Separates queries in DF into Easy, Medium and Hard by using number words present in query.
    :param df: input dataframe
    :param difficulty: can be easy, medium or hard
    :param numSamples: number of samples from dataset
    :return: subset of the data
    """
    if difficulty == "easy":
        return df[df["total_length"].between(10, 100)].iloc[:numSamples]
    elif difficulty == "medium":
        return df[df["total_length"].between(101, 200)].iloc[:numSamples]
    elif difficulty == "hard":
        return df[df["total_length"].between(201, 800)].iloc[:numSamples]


def create_and_save_datasets(df: pd.DataFrame, subset: str, train_ratio: int = 0.8, val_ratio: int = 0.1,
                             test_ratio: int = 0.1) -> datasets.DatasetDict:
    """
    Splitting Dataset into Training, Validation and Test files.
    :param df: dataframe
    :param subset: Can be easy, medium or hard
    :param train_ratio: number of train samples
    :param val_ratio: number of val samples
    :param test_ratio: number of test samples
    :return: dataset with train, test and validation splits
    """
    seed = 123
    # remove total_length column because we don't need it anymore
    if "total_length" in df.columns:
        df = df.drop(columns=["total_length"])
    dataset = datasets.Dataset.from_pandas(df, preserve_index=False)

    # split into training and "the rest"
    train_val_test = dataset.train_test_split(train_size=train_ratio, seed=seed)

    # split "the rest" into validation and testing
    val_test = train_val_test["test"].train_test_split(
        test_size=test_ratio / (test_ratio + val_ratio), seed=seed
    )

    dataset = datasets.DatasetDict(
        {
            "train": train_val_test["train"],
            "valid": val_test["train"],
            "test": val_test["test"],
        }
    )
    dataset_name = f"text-to-sql-v1-{subset}"
    dataset.save_to_disk(dataset_name)
    return dataset


def create_dataset(subset: str, split: bool = False, num_samples: int = 10000) -> datasets.DatasetDict:
    """
    Creates the dataset or if dataset already exists then loads it from disk.
    :param split: boolean var to split data into easy, medium and hard
    :param subset: Can be easy, medium or hard
    :param num_samples: number of samples in dataset
    :return: subset of dataset
    """

    try:
        datasets.load_from_disk(f"text-to-sql-v1-{subset}")
    except FileNotFoundError:
        dataset = datasets.load_dataset("Clinton/Text-to-sql-v1")
        dataset = dataset["train"]
        dataset = dataset.remove_columns(["text", "source"])
        df = add_length_column(dataset)
        if split:
            df = filter_by_total_length(df, subset, num_samples)
        return create_and_save_datasets(df, subset)
