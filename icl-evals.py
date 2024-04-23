import torch
from torch.utils.data import DataLoader

from multiprocessing import cpu_count

from tqdm import tqdm
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
from composer.models import write_huggingface_pretrained_from_composer_checkpoint

from torchmetrics import Metric

import os
from collections import defaultdict

import schedulefree
from torchinfo import summary

reproducibility.configure_deterministic_mode()
reproducibility.seed_all(42)


def icl_tokenize(tokenizer, context_window=4096):
    def icl_tokenize_inner(examples):
        context_indices = tokenizer.encode(
            examples["context"],
        )
        continuation_indices = tokenizer.encode(
            examples["continuation"],
        )

        return {
            "context_indices": context_indices,
            "continuation_indices": continuation_indices,
        }

    return icl_tokenize_inner


def icl_collate_fn(tokenizer):
    def icl_collate_fn_inner(examples):
        context_indices = [example["context_indices"] for example in examples]
        continuation_indices = [example["continuation_indices"] for example in examples]

        context_indices = torch.nn.utils.rnn.pad_sequence(
            torch.tensor(context_indices, dtype=torch.long),
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )

        continuation_indices = torch.nn.utils.rnn.pad_sequence(
            torch.tensor(continuation_indices, dtype=torch.long),
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )

        return {
            "context_indices": context_indices,
            "continuation_indices": continuation_indices,
        }

    return icl_collate_fn_inner


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
    batch_size = 16
    run_name = "icl-run-9"
    eval_interval = "6000ba"
    learning_rate = 1e-4

    tokenizer_file = f"{datafolder}/overall-tokenizer.json"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_file, max_len=context_window, padding_side="right", truncation=True
    )

    print(f"tokenizer length: {len(tokenizer)}")

    icl_dataset = load_dataset(
        "json",
        data_files={
            "eval": f"{datafolder}/eval-icl/eval_suits_4_cards_15_decks_1.jsonl"
        },
        cache_dir=f"{datafolder}/.cache",
    ).map(icl_tokenize(tokenizer, context_window=context_window))["eval"]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=context_window
    )

    icl_dataloader = DataLoader(
        icl_dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=icl_collate_fn(tokenizer),
        pin_memory=True,
        num_workers=cpu_count(),
    )

    write_huggingface_pretrained_from_composer_checkpoint(
        f"checkpoints/{run_name}/ep1-ba37679-rank0.pt",
        f"checkpoints/{run_name}/hf",
    )

    model = LlamaForCausalLM.from_pretrained(f"checkpoints/{run_name}/hf")
    model.to(device)
    model.eval()

    space_token_id = tokenizer.convert_tokens_to_ids(" ")
    star_token_id = tokenizer.convert_tokens_to_ids("*")
    print(f"space token ID: {space_token_id}, star token ID: {star_token_id}")

    stop_token_counts = defaultdict(int)
    generated_star_counts = defaultdict(int)
    correct_predictions = 0
    total_predictions = 0
    star_differences = []

    for data in tqdm(iter(icl_dataloader)):
        generated_token_ids = []
        input_ids = data["context_indices"].to(device)
        stopped_at = input_ids.size(1)

        while True:
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                generated_token_id = torch.argmax(outputs.logits[0, -1]).item()
                generated_token_ids.append(generated_token_id)

                if (
                    generated_token_id != space_token_id
                    and generated_token_id != star_token_id
                ):
                    stopped_at = generated_token_id
                    break

                input_ids = torch.cat(
                    (input_ids, torch.tensor([[generated_token_id]]).to(device)), dim=1
                )

        # Update stop token counts
        stop_token_counts[stopped_at] += 1

        # Count the number of stars in the generated token IDs
        generated_stars = generated_token_ids.count(star_token_id)
        generated_star_counts[generated_stars] += 1

        # Count the number of stars in the continuation token IDs
        actual_stars = data["continuation_indices"][0].tolist().count(star_token_id)

        if generated_stars == actual_stars:
            correct_predictions += 1
        total_predictions += 1

        star_difference = abs(generated_stars - actual_stars)
        star_differences.append(star_difference)

    # Compute stop token statistics
    for token_id, count in stop_token_counts.items():
        percentage = count / total_predictions * 100
        print(f"Stopped at token ID {token_id}: {percentage:.2f}%")

    # Compute generated star length statistics
    for star_length, count in generated_star_counts.items():
        percentage = count / total_predictions * 100
        print(f"Generated {star_length} stars: {percentage:.2f}% of the time")

    # Compute accuracy and mean star difference
    accuracy = correct_predictions / total_predictions
    mean_star_difference = sum(star_differences) / len(star_differences)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean Star Difference: {mean_star_difference:.2f}")
