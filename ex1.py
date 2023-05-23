from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EvalPrediction,
    Trainer,
    set_seed,
    TrainingArguments
)
from datasets import load_dataset
from evaluate import load
import numpy as np
import time
import os
import sys
# import wandb
#
# WANDB_PROJECT = "anlp_ex1"

OUTPUT_DIR = "/output"

MODELS = ['bert-base-uncased',
          'roberta-base',
          'google/electra-base-generator']

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    num_seeds: int = int(sys.argv[1])
    num_train_samples: int = int(sys.argv[2])
    num_validation_samples: int = int(sys.argv[3])
    num_samples_to_predict: int = int(sys.argv[4])

    # ### WANDB SETUP ###
    #
    # # set the wandb project where this run will be logged
    # os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    #
    # # save your trained model checkpoint to wandb
    # os.environ["WANDB_LOG_MODEL"] = "true"
    #
    # # turn off watch to log faster
    # os.environ["WANDB_WATCH"] = "false"
    #
    # wandb.login()
    #
    # ### END WANDB SETUP ###


    # load dataset
    dataset = load_dataset("sst2")
    if num_train_samples == -1:
        train_data = dataset["train"]
    else:
        train_data = dataset["train"].select(range(num_train_samples))
    if num_validation_samples == -1:
        validation_data = dataset["validation"]
    else:
        validation_data = dataset["validation"].select(range(num_validation_samples))
    if num_samples_to_predict == -1:
        test_data = dataset["test"]
    else:
        test_data = dataset["test"].select(range(num_samples_to_predict))

    models_to_accuracies = {}
    train_start_time = time.time()

    for model_to_load in MODELS:
        print(f"\n\n\n..........\nSTARTING MODEL {model_to_load}\n.........\n\n\n")
        models_to_accuracies[model_to_load] = []

        for seed in range(num_seeds):
            accuracy, trainer, preprocess_function = \
                finetune_model(model_to_load, seed, train_data, validation_data)

            models_to_accuracies[model_to_load].append((accuracy, trainer, preprocess_function))

    train_end_time = time.time()

    best_mean = 0
    best_model_name = MODELS[0]
    model_to_mean_std = {}

    for model_name, accuracies_to_models in models_to_accuracies.items():
        accuracies, _, __ = zip(*accuracies_to_models)
        accuracies = np.array(accuracies)
        mean, std = np.mean(accuracies), np.std(accuracies)
        model_to_mean_std[model_name] = (accuracies, mean, std)

        if mean > best_mean:
            best_mean = mean
            best_model_name = model_name

    print(f"\n\n\n.............\nTOP MODEL: {best_model_name}")
    best_seed = int(np.argmax(model_to_mean_std[best_model_name][0]))
    print(f"TOP SEED: {best_seed}\n.............")
    set_seed(best_seed)
    best_trainer = models_to_accuracies[best_model_name][best_seed][1]
    preprocess_function = models_to_accuracies[best_model_name][best_seed][2]


    prediction_start_time = time.time()

    # map and freeze
    test_ds = test_data.map(preprocess_function, batched=True)
    test_ds = test_ds.remove_columns("label")
    best_trainer.model.eval()

    test_set_predictions_proba = []
    for i in range(len(test_ds)):
        example = test_ds.select(range(i, i + 1))
        predictions_proba = best_trainer.predict(example).predictions
        test_set_predictions_proba.append(predictions_proba)

    test_set_predictions_proba = np.concatenate(test_set_predictions_proba, axis=0)
    predictions = np.argmax(test_set_predictions_proba, axis=1)

    prediction_end_time = time.time()

    print("Writing result files...")
    write_predictions_file(predictions, test_ds)
    write_res_file(model_to_mean_std,
                       train_end_time - train_start_time,
                       prediction_end_time - prediction_start_time)


def write_predictions_file(predictions, test_ds):
    predicts_txt = ""
    for i, prediction in enumerate(predictions):
        predicts_txt += f"{test_ds[i]['sentence']}###{prediction}\n"

    with open(os.path.join(OUTPUT_DIR, "predictions.txt"), "w") as f:
        f.write(predicts_txt)


def write_res_file(model_to_mean_std, train_time, predict_time):
    res_txt = ""
    for model_name, (accuracies, mean, std) in model_to_mean_std.items():
        res_txt += f"{model_name},{mean} +- {std}\n"
    res_txt += "----\n"
    res_txt += f"train time,{round(train_time, 2)}\n"
    res_txt += f"predict time,{round(predict_time, 2)}"

    with open(os.path.join(OUTPUT_DIR, "res.txt"), "w") as f:
        f.write(res_txt)


def finetune_model(model_to_load, seed, train_data, validation_data):
    run_name = f"{model_to_load.replace('/', '-')}_seed{seed}"
    run_path = os.path.join(OUTPUT_DIR, run_name)
    # wandb.init(name=run_name, dir=OUTPUT_DIR, project=WANDB_PROJECT, reinit=True)

    print(f"\n\n\n............\nSTARTING SEED {seed} ON MODEL {model_to_load}.......")
    # load model and tokenizer
    config = AutoConfig.from_pretrained(model_to_load)
    tokenizer = AutoTokenizer.from_pretrained(model_to_load)

    run_finetune = not os.path.exists(os.path.join(run_path, 'pytorch_model.bin'))
    model_src = model_to_load if run_finetune else run_path
    model = AutoModelForSequenceClassification.from_pretrained(model_src, config=config)

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

    training_args = TrainingArguments(output_dir=run_path,
                                      # report_to='wandb',
                                      run_name=run_name,
                                      save_strategy="no",
                                      seed=seed
                                      )

    # init trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    if run_finetune:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model(run_path)

        trainer.save_metrics("train", metrics)
        trainer.log_metrics("train", metrics)
        trainer.save_state()
    # validate
    trainer.model.eval()
    metrics = trainer.evaluate(eval_dataset=validation_data)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    # wandb.finish()
    return metrics['eval_accuracy'], trainer, preprocess_function


if __name__ == '__main__':
    main()