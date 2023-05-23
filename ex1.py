from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments
from datasets import load_dataset


@dataclass
class runParams:
    num_seeds: int = field(default=0, metadata={"help": "number of seeds to train with"})
    num_train_samples: int = field(default=-1, metadata={"help": "number of train examples to use"})
    num_validation_samples: int = field(default=-1, metadata={"help": "number of validation examples to use"})
    num_samples_to_predict: int = field(default=-1, metadata={"help": "number of samples to predict a sentiment for"})

def main():
    parser = HfArgumentParser((runParams))
    my_params = parser.parse_args_into_dataclasses()[0]

    # load data
    dataset = load_dataset("sst2")
    train_data = dataset['train'] if my_params.num_train_samples == -1 \
        else dataset['train'].select(range(my_params.num_train_samples))
    validation_data = dataset['validation'] if my_params.num_validation_samples == -1\
        else dataset['validation'].select(range(my_params.num_validation_samples))
    test_data = dataset['test'] if my_params.num_samples_to_predict == -1 \
        else dataset['test'].select(range(my_params.num_samples_to_predict))

if __name__ == '__main__':
    main()
print(1)