from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments
from datasets import load_dataset

@dataclass
class runParams:
    num_seeds: int = field(metadata={"help": "number of seeds to train with"})
    num_train_samples: int
    num_validation_samples: int
    num_samples_to_predict: int

def main():
    parser = HfArgumentParser((runParams, TrainingArguments))
    my_params, training_args = parser.parse_args_into_dataclasses()

    dataset = load_dataset("sst2")


if __name__ == '__main__':
    main()
print(1)