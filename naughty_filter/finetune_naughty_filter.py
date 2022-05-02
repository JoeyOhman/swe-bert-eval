import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser, \
    DataCollatorWithPadding, TrainingArguments, Trainer, AutoConfig
from datasets import load_dataset, load_metric

from utils import set_seed, get_device, ModelArguments, DataTrainingArguments


def load_tokenizer_model(model_args):
    # tokenizer = BertTokenizerFast.from_pretrained(model_args.model_name_or_path, do_lower_case=False)
    # tokenizer = BertTokenizerFast.from_pretrained("KB/bert-base-swedish-cased", do_lower_case=False)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    # config = AutoConfig.from_pretrained("KB/bert-base-swedish-cased")
    config.num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)
    # model = AutoModelForSequenceClassification.from_config(config)
    model.to(device)
    return tokenizer, model


def pre_process_data(tokenizer, dataset_split, max_len):
    dataset_split = dataset_split.map(
        lambda sample: tokenizer(sample['text'], truncation=True, max_length=max_len),
        batched=True, num_proc=4)

    dataset_split = dataset_split.map(lambda sample: {'labels': sample['label']}, batched=True, num_proc=4)

    dataset_split.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    dataset_split = dataset_split.remove_columns(['text', 'label'])
    # dataloader = torch.utils.data.DataLoader(dataset_split, batch_size=BATCH_SIZE)
    return dataset_split


def create_naughty_dataset(data_args, tokenizer, max_len):
    print("Loading and pre-processing dataset...")
    dataset = load_dataset('json', data_files={
        'train': "data/naughty_dataset_train.jsonl",
        'validation': "data/naughty_dataset_val.jsonl",
        'test': "data/naughty_dataset_test.jsonl",
    })

    train_ds, val_ds, test_ds = dataset['train'], dataset['validation'], dataset['test']

    df = data_args.data_fraction
    assert 0.0 < df <= 1.0, "data_fraction must be in range (0, 1]"
    if df < 1.0:
        train_ds = train_ds.select(range(int(len(train_ds) * df)))
        val_ds = val_ds.select(range(int(len(val_ds) * df)))
        test_ds = test_ds.select(range(int(len(test_ds) * df)))

    train_ds = pre_process_data(tokenizer, train_ds, max_len)
    val_ds = pre_process_data(tokenizer, val_ds, max_len)
    test_ds = pre_process_data(tokenizer, test_ds, max_len)

    train_ds_lens = [sample['input_ids'].shape[0] for sample in train_ds]
    print("Train ds, Max len:", max(train_ds_lens))
    print("Train ds, Mean len:", np.mean(train_ds_lens))

    return train_ds, val_ds, test_ds


def init_trainer(training_args, model, tokenizer, train_ds, val_ds):

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    metric = load_metric('accuracy')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return training_args, trainer


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer, model = load_tokenizer_model(model_args)

    max_seq_len = min(model.config.max_position_embeddings, data_args.max_input_length)
    train_ds, val_ds, test_ds = create_naughty_dataset(data_args, tokenizer, max_seq_len)

    training_args, trainer = init_trainer(training_args, model, tokenizer, train_ds, val_ds)

    trainer.train()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    predictions, labels, metrics = trainer.predict(test_ds)
    # predictions = np.argmax(predictions, axis=1)
    trainer.log_metrics("test", metrics)


if __name__ == '__main__':
    set_seed(42)
    device = get_device()
    main()
