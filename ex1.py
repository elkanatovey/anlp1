from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    EvalPrediction,
    Trainer,
    set_seed
)
from datasets import load_dataset
from evaluate import load
import numpy as np

MODELS = ['bert-base-uncased',
          'roberta-base',
          'google/electra-base-generator']
@dataclass
class runParams:
    num_seeds: int = field(default=0, metadata={"help": "number of seeds to train with"})
    num_train_samples: int = field(default=-1, metadata={"help": "number of train examples to use"})
    num_validation_samples: int = field(default=-1, metadata={"help": "number of validation examples to use"})
    num_samples_to_predict: int = field(default=-1, metadata={"help": "number of samples to predict a sentiment for"})

def main():
    parser = HfArgumentParser((runParams))
    my_params = parser.parse_args_into_dataclasses()[0]

    # load dataset
    dataset = load_dataset("sst2")
    train_data = dataset['train'] if my_params.num_train_samples == -1 \
        else dataset['train'].select(range(my_params.num_train_samples))
    validation_data = dataset['validation'] if my_params.num_validation_samples == -1\
        else dataset['validation'].select(range(my_params.num_validation_samples))
    test_data = dataset['test'] if my_params.num_samples_to_predict == -1 \
        else dataset['test'].select(range(my_params.num_samples_to_predict))

    for model_to_load in MODELS:

        # load model and tokenizer
        config = AutoConfig.from_pretrained(model_to_load)
        tokenizer = AutoTokenizer.from_pretrained(model_to_load)
        model = AutoModel.from_pretrained(model_to_load, config=config)

        # tokenize train data
        def preprocess_function(examples):
            return tokenizer(examples['sentence'], truncation=True)

        train_data = train_data.map(preprocess_function, batched=True)
        validation_data = validation_data.map(preprocess_function, batched=True)

        # define metric
        metric = load("accuracy")

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            return metric.compute(predictions=preds, references=p.label_ids)

        # init trainer
        trainer = Trainer(
            model=model,
            # args=training_args,
            train_dataset=train_data,
            eval_dataset=validation_data,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )

        #train
        train_result = trainer.train()







    print(2)


if __name__ == '__main__':
    main()
print(1)