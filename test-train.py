import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from multiprocessing import cpu_count
from transformers import PreTrainedTokenizerFast

from transformers import LlamaConfig, LlamaForCausalLM
from transformers import DataCollatorForLanguageModeling
from composer.utils import reproducibility
from composer.metrics.nlp import LanguagePerplexity
from composer.models import HuggingFaceModel
from composer import Trainer

import os

import schedulefree

reproducibility.configure_deterministic_mode()
reproducibility.seed_all(42)


class BlackjackDataset(Dataset):
    def __init__(self, tokenizer_file, data_file, context_window=8192):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_file)

        with open(data_file, "r") as f:
            self.data = self.tokenizer.encode(f.read())
        self.context_window = context_window

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError

        start_at_idx = (idx - self.context_window) if idx > self.context_window else 0

        inputs = self.data[
            start_at_idx : idx + 1
        ]  # Include the target token in the input sequence
        inputs = torch.tensor(inputs)

        return inputs


if __name__ == "__main__":
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["NODE_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    torch.distributed.init_process_group()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    context_window = 1024
    train_dataset = BlackjackDataset(
        "data/blackjack_transcripts-tokenizer.json",
        "data/blackjack_transcripts.txt",
        context_window=context_window,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=train_dataset.tokenizer, mlm=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=96,
        pin_memory=True,
        collate_fn=data_collator,
        persistent_workers=True,
        num_workers=cpu_count(),
    )

    config = LlamaConfig(
        vocab_size=train_dataset.tokenizer.vocab_size,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=6,
        hidden_size=128,
        intermediate_size=256,
        max_position_embeddings=context_window,
        tie_word_embeddings=True,
    )

    model = HuggingFaceModel(
        LlamaForCausalLM(config),
        tokenizer=train_dataset.tokenizer,
        metrics=[LanguagePerplexity(ignore_index=-100)],
        use_logits=True,
    )
    model.to(device)
    model.train()

    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=3e-4)
    print(model)

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=train_dataloader,
        max_duration="1ep",
        save_folder="checkpoints",
        save_filename="ep{epoch}.pt",
        save_interval="1500ba",
        save_overwrite=False,
        device_train_microbatch_size="auto",
        device="gpu",
        run_name="test-train",
        autoresume=True,
        precision="amp_bf16",
    )

    trainer.fit()
