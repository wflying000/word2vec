import os
import sys
import time
import torch
import pickle
from torch.utils.data import DataLoader

from trainer import Trainer
from model import CBOW, SkipGram
from dataset import load_data, create_contexts_target, NegativeSampler, CBOWDataset, SkipGramDataset


def train_cbow():
    filepath = "../data/ptb.train.txt"
    window_size = 5
    embed_dim = 100
    batch_size = 100
    num_epochs = 10
    negative_sample_size = 5
    learning_rate = 1e-3
    now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    outputs_dir = f"../outputs/cbow-{now_time}"
    os.makedirs(outputs_dir, exist_ok=True)
    device = torch.device("cuda")

    corpus, word2id, id2word = load_data(filepath)
    contexts, targets = create_contexts_target(corpus, window_size) 
    vocab_size = len(word2id)
    corpus_info = {
        "word2id": word2id,
        "id2word": id2word,
    }
    save_path = os.path.join(outputs_dir, "corpus_info.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(corpus_info, f)
    

    negative_sampler = NegativeSampler(corpus, negative_sample_size)
    train_dataset = CBOWDataset(
        contexts=contexts,
        targets=targets,
        negative_sampler=negative_sampler,
    )
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.generate_batch,
        num_workers=16,
        pin_memory=True,
    )

    model = CBOW(vocab_size, embed_dim)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        outputs_dir=outputs_dir,
        num_epochs=num_epochs,
        device=device,
    )

    trainer.train()


def train_skipgram():
    filepath = "../data/ptb.train.txt"
    window_size = 5
    embed_dim = 100
    batch_size = 100
    num_epochs = 10
    negative_sample_size = 5
    learning_rate = 1e-3
    now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    outputs_dir = f"../outputs/skipgram-{now_time}/"
    os.makedirs(outputs_dir, exist_ok=True)
    device = torch.device("cuda")

    corpus, word2id, id2word = load_data(filepath)
    contexts, targets = create_contexts_target(corpus, window_size) 
    vocab_size = len(word2id)

    corpus_info = {
        "corpus": corpus,
        "word2id": word2id,
        "id2word": id2word,
        "contexts": contexts,
        "targets": targets,
    }

    with open("../data/corpus_info.pkl", "wb") as f:
        pickle.dump(corpus_info, f)

    negative_sampler = NegativeSampler(corpus, negative_sample_size)

    train_dataset = SkipGramDataset(
        contexts=contexts,
        centers=targets,
        negative_sampler=negative_sampler,
    )
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.generate_batch,
        num_workers=16,
        pin_memory=True,
    )

    model = SkipGram(vocab_size, embed_dim)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        outputs_dir=outputs_dir,
        num_epochs=num_epochs,
        device=device,
    )

    trainer.train()

if __name__ == "__main__":
    os.chdir(sys.path[0])
    train_cbow()
    train_skipgram()





