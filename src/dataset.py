import math
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


def get_words(filepath):
    with open(filepath) as f:
        text = f.read()
    text1 = text.replace('\n', "<eos>")
    text2 = text1.strip()
    words = text2.split()
    return words


def load_data(filepath):
    words = get_words(filepath)
    word2id = {}
    id2word = {}
    for word in words:
        if word not in word2id:
            idx = len(word2id)
            word2id[word] = idx
            id2word[idx] = word
    
    corpus = [word2id[word] for word in words]

    return corpus, word2id, id2word

def create_contexts_target(corpus, window_size):

    targets = corpus[window_size: -window_size]
    contexts = []
    total = len(corpus) - window_size - window_size

    for idx in tqdm(range(window_size, len(corpus) - window_size), total=total, leave=False):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return contexts, targets

class NegativeSampler:
    def __init__(self, corpus, sample_size, power=0.75):
        self.corpus = corpus
        self.sample_size = sample_size

        id2count = {}
        for word_id in corpus:
            if word_id not in id2count:
                id2count[word_id] = 0
            id2count[word_id] += 1

        total = 0
        for word_id, count in id2count.items():
            new_count = math.pow(count, power)
            id2count[word_id] = new_count
            total += new_count

        self.vocab_size = len(id2count)

        self.id2prob = np.zeros(self.vocab_size)
        for word_id, count in id2count.items():
            self.id2prob[word_id] = count / total
        
    def get_negative_sample(self, target):
        batch_size = len(target)

        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

        for i in range(batch_size):
            p = self.id2prob.copy()
            target_idx = target[i]
            p[target_idx] = 0
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)

        return negative_sample


class CBOWDataset(Dataset):
    def __init__(self, contexts, targets, negative_sampler):
        self.contexts = contexts
        self.targets = targets
        self.negative_sampler = negative_sampler

    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, idx):
        contexts = self.contexts[idx]
        targets = [self.targets[idx]]
        negative_samples = self.negative_sampler.get_negative_sample(targets)
        targets += [x for x in negative_samples[0]]
        labels = [1] + [0 for _ in range(len(negative_samples[0]))]

        item = {
            "contexts": contexts,
            "targets": targets,
            "labels": labels,
        }

        return item
    
    def generate_batch(self, item_list):
        contexts = [x["contexts"] for x in item_list]
        targets = [x["targets"] for x in item_list]
        labels = [x["labels"] for x in item_list]

        outputs = {
            "contexts": torch.LongTensor(contexts),
            "targets": torch.LongTensor(targets),
            "labels": torch.LongTensor(labels),
        }

        return outputs
    

class SkipGramDataset(Dataset):
    def __init__(self, contexts, centers, negative_sampler):
        self.contexts = contexts
        self.centers = centers
        self.negative_sampler = negative_sampler
    
    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, idx):
        center = self.centers[idx]
        context = self.contexts[idx]
        negative_samples = self.negative_sampler.get_negative_sample(context)
        negative_samples = negative_samples.reshape(-1).tolist()
        label = [1] * len(context) + [0] * len(negative_samples)
        context_negative_samples = context + negative_samples

        item = {
            "center": center,
            "context": context_negative_samples,
            "label": label,
        }
        return item
    
    def generate_batch(self, item_list):
        center_ids = [x["center"] for x in item_list]
        context_ids = [x["context"] for x in item_list]
        labels = [x["label"] for x in item_list]

        outputs = {
            "center_ids": torch.LongTensor(center_ids),
            "context_ids": torch.LongTensor(context_ids),
            "labels": torch.LongTensor(labels),
        }

        return outputs


def test():
    import os
    import sys
    from torch.utils.data import DataLoader
    os.chdir(sys.path[0])

    filepath = "../data/ptb.train.txt"
    window_size = 5
    sample_size = 3
    corpus, word2id, id2word = load_data(filepath)
    contexts, targets = create_contexts_target(corpus, window_size)
    negative_sampler = NegativeSampler(corpus, sample_size)

    cbow_dataset = CBOWDataset(contexts, targets, negative_sampler)

    cbow_dataloader = DataLoader(
        dataset=cbow_dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=cbow_dataset.generate_batch,
    )

    # for batch in tqdm(cbow_dataloader, total=len(cbow_dataloader)):
    #     pass

    sg_dataset = SkipGramDataset(contexts, targets, negative_sampler)
    sg_dataloader = DataLoader(
        dataset=sg_dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=sg_dataset.generate_batch,
    )

    for batch in tqdm(sg_dataloader, total=len(sg_dataloader)):
        pass


if __name__ == "__main__":
    test()






        




    
