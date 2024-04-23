import torch
from torch.utils.data import DataLoader

from multiprocessing import cpu_count

from transformers import PreTrainedTokenizerFast
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import datasets

from composer.utils import reproducibility
from composer.metrics.nlp import LanguagePerplexity, InContextLearningQAAccuracy
from composer.models import HuggingFaceModel
from composer import Trainer
from composer.core import Evaluator

from composer.loggers import FileLogger, TensorboardLogger, NeptuneLogger
from composer import Callback, Event, Logger, State

import os

import schedulefree
from torchinfo import summary

reproducibility.configure_deterministic_mode()
reproducibility.seed_all(42)


def create_sliding_windows(tokenizer, context_window=1024):
    def create_sliding_windows_inner(examples):
        input_ids = []
        labels = []

        for text in examples["text"]:
            encoded_text = tokenizer.encode(text)

            for i in range(len(encoded_text) - context_window):
                input_ids.append(encoded_text[i : i + context_window])
                labels.append(encoded_text[i + context_window])

        return {"input_ids": input_ids, "labels": labels}

    return create_sliding_windows_inner


if __name__ == "__main__":
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["NODE_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    datasets.config.IN_MEMORY_MAX_SIZE = 8e9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datafolder = "data-generated-icl"
    context_window = 4096
    batch_size = 8
    run_name = "icl-run-9"
    eval_interval = "6000ba"
    learning_rate = 1e-4

    tokenizer_file = f"{datafolder}/overall-tokenizer.json"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_file, padding_side="left"
    )

    print(f"tokenizer length: {len(tokenizer)}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=context_window
    )

    dataset = load_dataset(
        "text",
        data_files={
            "train": [f"{datafolder}/train/*.txt"],
            "eval": [f"{datafolder}/eval/*.txt"],
        },
        sample_by="document",
        keep_linebreaks=True,
        cache_dir=f"{datafolder}/.cache",
    ).map(
        create_sliding_windows(
            tokenizer=tokenizer,
            context_window=context_window,
        ),
        batched=True,
        batch_size=1,
        num_proc=1,
        remove_columns=["text"],
    )

    train_dataloader = DataLoader(
        dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        pin_memory=False,
        collate_fn=data_collator,
        prefetch_factor=int(batch_size / 8),
        num_workers=cpu_count(),
    )

    eval_dataloader = DataLoader(
        dataset["eval"],
        shuffle=False,
        batch_size=batch_size,
        pin_memory=False,
        collate_fn=data_collator,
        prefetch_factor=int(batch_size / 8),
        num_workers=cpu_count(),
    )

    ppl_eval = Evaluator(
        label="labels",
        dataloader=eval_dataloader,
        metric_names=["LanguagePerplexity"],
    )

    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=12,
        hidden_size=512,
        intermediate_size=512,
        tie_word_embeddings=True,
        rms_norm_eps=1e-5,
        rope_theta=500000,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.get_vocab()["BeginSession"],
        eos_token_id=tokenizer.get_vocab()["EndSession"],
        max_position_embeddings=context_window,
        use_cache=True,
    )

    model = HuggingFaceModel(
        LlamaForCausalLM(config),
        tokenizer=tokenizer,
        metrics=[LanguagePerplexity(ignore_index=-100)],
        use_logits=True,
    )
    model.to(device)
    model.train()

    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=learning_rate)
    summary(model)

    torch.distributed.init_process_group()

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=[ppl_eval],
        eval_interval=eval_interval,
        max_duration="1ep",
        save_folder=f"checkpoints/{run_name}",
        save_interval=eval_interval,
        save_overwrite=False,
        device_train_microbatch_size="auto",
        device="gpu",
        run_name=run_name,
        autoresume=True,
        precision="amp_fp16",
        console_log_interval=eval_interval,
        loggers=[
            FileLogger(f"checkpoints/{run_name}_logs.txt"),
            TensorboardLogger(),
            # NeptuneLogger(
            #     project="blackjack-synthetic",
            #     name=run_name,
            #     # description="Test run #6",
            #     source_files="*.py",
            #     upload_artifacts=True,
            #     git_ref=True,
            #     dependencies="infer",
            #     mode="sync",
            # ),
        ],
    )

    trainer.fit()
