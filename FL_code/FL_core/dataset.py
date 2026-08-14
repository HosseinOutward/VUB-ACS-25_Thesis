from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import SVHN, CIFAR10, CIFAR100, STL10, MNIST, Imagenette, EuroSAT, Food101
from torchvision.transforms import v2 as transforms
from transformers import AutoTokenizer

from .models import BERT_CHECKPOINT, BERT_SEQUENCE_LENGTH, DISTILBERT_CHECKPOINT

if TYPE_CHECKING:
    from FL_code.run_fl import FLConfig


# --- Shared-memory dataset --- #
class SharedTensorDataset(Dataset):
    """Zero-copy dataset backed by shared-memory tensors."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X, self.y = X, y

        # Ensure tensors are in shared memory
        assert self.y.is_shared()
        assert self.X.is_shared()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# --- Dataset registry --- #
# Dataset configurations: (mean, std)
DATASET_CONFIG: dict[str, tuple[tuple[float, float, float],...]] = {
    'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    'CIFAR100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    'STL10': ((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
    'MNIST': ((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    'IMAGENETTE': ((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
    'EUROSAT': ((0.3444, 0.3803, 0.4078), (0.2027, 0.1369, 0.1156)),
    'FEMNIST': ((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    'FOOD101': ((0.5450, 0.4435, 0.3436), (0.2695, 0.2725, 0.2764)),
    'TINYIMAGENET': ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
    'SYNTHETIC': ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
}
# Text sets, tokenized rather than normalized, so they carry no mean/std. Each is tokenized once,
# with one checkpoint's vocabulary, so it can only be paired with a model that reads that vocabulary
# - `run_fl.FLConfig` enforces the pairing. The two uncased WordPiece vocabularies here happen to be
# identical, so a mismatch would produce plausible-looking garbage rather than an error.
TEXT_DATASETS: Final[frozenset[str]] = frozenset(
    {'20NEWS', 'AGNEWS', 'SENT140', 'DBPEDIA', 'YELP5'})
TEXT_CHECKPOINT: Final[dict[str, str]] = {
    '20NEWS': BERT_CHECKPOINT,
    'AGNEWS': DISTILBERT_CHECKPOINT,
    'SENT140': DISTILBERT_CHECKPOINT,
    'DBPEDIA': DISTILBERT_CHECKPOINT,
    'YELP5': DISTILBERT_CHECKPOINT,
}
# Datasets whose samples carry a natural client id, which the non-IID partition groups on.
GROUP_FIELD: Final[dict[str, str]] = {'FEMNIST': 'writer_id', 'SENT140': 'user'}
# The canonical stanfordnlp/sentiment140 is script-backed, which `datasets` 5 no longer loads;
# this mirror is plain CSV and, unlike most mirrors, keeps the `user` column the partition needs.
SENT140_REPO: Final[str] = "bdanko/sentiment140"
# Kept per split: the busiest users, so one-user-per-client clients are not starved.
SENT140_USERS: Final[int] = 600
# AG News ships 120k training rows, which at 66M parameters costs ten minutes a round for no gain -
# DistilBERT already clears 93% in one. Cut to Sent140's scale so the two text setups cost the same
# and differ only in data and partition, which is the comparison being bought.
AGNEWS_SAMPLES: Final[int] = 48_000
# DBpedia ships 560k rows. Cut to AG News' size so the two topic-classification setups cost the same
# and differ only in the task: news category against DBpedia's ontology class.
DBPEDIA_SAMPLES: Final[int] = 48_000
DBPEDIA_REPO: Final[str] = "fancyzhx/dbpedia_14"
# The one text task in the sweep that pretrained DistilBERT cannot solve in a single federated
# round. AG News and DBpedia both arrive at their final accuracy in round 0 - DBpedia at 98.98% -
# so all four measurement windows sample the same converged model and the phase variable this
# experiment is built around never moves. Yelp's five star ratings are ordinal with genuinely
# overlapping neighbours, and a fully fine-tuned DistilBERT tops out near 65%, so the run spends
# its rounds actually training. Same 48k budget as the other two, so cost is comparable.
YELP5_SAMPLES: Final[int] = 48_000
YELP5_REPO: Final[str] = "Yelp/yelp_review_full"
TINYIMAGENET_REPO: Final[str] = "zh-plus/tiny-imagenet"
# Language modelling rather than classification: the target is the next token, so `y` is a sequence
# and the "class count" is the vocabulary. Word-level with the corpus's own <unk>, as the benchmark
# is normally reported, and split into independent blocks - the LSTM carries no state across them.
WIKITEXT2_REPO: Final[str] = "Salesforce/wikitext"
WIKITEXT2_CONFIG: Final[str] = "wikitext-2-v1"
WIKITEXT2_BLOCK: Final[int] = 128
WIKITEXT2_UNK: Final[str] = "<unk>"
DATASET_NAMES: tuple[str, ...] = (
    *DATASET_CONFIG, '20NEWS', 'AGNEWS', 'SENT140', 'DBPEDIA', 'YELP5', 'WIKITEXT2')

# EuroSAT and FEMNIST ship no official split, so one is fixed here, and AG News is subsampled with
# it. The seed is deliberately not `cfg.seed`: the train and test calls are separate, and they must
# partition the same source identically no matter what seed the run uses, or the halves would
# overlap.
SPLIT_SEED: Final[int] = 12_345
EUROSAT_TRAIN_FRACTION: Final[float] = 0.8


# --- Decoding, resampling and grouping helpers --- #
def _decode_images(
    dataset: Any, indices: list[int], image_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the given rows of a PIL-backed torchvision dataset to one normalized image tensor."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.PILToTensor(),
    ])
    images = torch.empty(len(indices), 3, image_size, image_size, dtype=torch.uint8)
    labels = torch.empty(len(indices), dtype=torch.long)
    for row, index in enumerate(indices):
        image, label = dataset[index]
        images[row] = transform(image.convert("RGB"))
        labels[row] = label
    return images.float().div_(255.0), labels


def _resize(images: torch.Tensor, image_size: int) -> torch.Tensor:
    """Resample decoded images when the setup asks for a resolution other than the native one."""
    if images.shape[-1] == image_size:
        return images
    return torch.nn.functional.interpolate(
        images, size=image_size, mode="bilinear", antialias=True, align_corners=False
    )


def _group_ids(values: np.ndarray) -> torch.Tensor:
    """Map arbitrary group labels to stable contiguous ids.

    Not `hash()`: string hashing is salted per process, so the same writer would be given a
    different id on every run and only the grouping, never the id itself, would be reproducible.
    """
    return torch.from_numpy(np.unique(values, return_inverse=True)[1].astype("int64"))


def _largest_groups(rows: list[int], groups: np.ndarray, max_groups: int) -> list[int]:
    """Keep the rows of the `max_groups` largest natural groups, order preserved.

    This is the single place the client pool is chosen, for the natural split and its IID twin
    alike, so the two provably draw on the same groups. It selects by size, which biases the pool
    towards the most prolific writers/users, so it prints what it kept: a run that takes 30 of
    3,057 writers is not the LEAF benchmark and should not be read as one.
    """
    assert len(rows) == len(groups), "each kept row needs exactly one group id"
    labels, counts = np.unique(groups, return_counts=True)
    assert len(labels) >= max_groups, (
        f"{len(labels)} natural groups cannot supply {max_groups} clients."
    )
    keep = set(labels[np.argsort(-counts)[:max_groups]].tolist())
    kept = [row for row, group in zip(rows, groups.tolist()) if group in keep]
    print(f"[Data] Keeping the {max_groups} largest of {len(labels)} natural groups: "
          f"{len(kept)}/{len(rows)} samples ({100 * len(kept) / len(rows):.1f}%).")
    return kept


def _wikitext_blocks(data_path: Path, is_train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """WikiText-2 as next-token blocks, with the vocabulary always built from the training split.

    Returns `[blocks, WIKITEXT2_BLOCK]` inputs and the same shape shifted one token left. The
    vocabulary is fixed by the training split and test tokens outside it map to <unk>, because a
    vocabulary that grew with the test set would let the model be scored on ids it never saw.
    """
    from datasets import load_dataset

    def tokens(split: str) -> list[str]:
        rows = load_dataset(WIKITEXT2_REPO, WIKITEXT2_CONFIG, split=split,
                            cache_dir=str(data_path / "hf"))["text"]
        return [word for line in rows for word in line.split()]

    train_words = tokens("train")
    vocabulary = {word: index for index, word in enumerate(sorted(set(train_words)))}
    assert WIKITEXT2_UNK in vocabulary, (
        f"{WIKITEXT2_CONFIG} should already carry {WIKITEXT2_UNK}; the raw config does not and "
        "cannot map unseen test tokens."
    )
    unknown = vocabulary[WIKITEXT2_UNK]
    words = train_words if is_train else tokens("test")
    ids = torch.tensor([vocabulary.get(word, unknown) for word in words], dtype=torch.long)

    # One token of overlap between input and target, so the last block keeps a full-length target.
    blocks = (len(ids) - 1) // WIKITEXT2_BLOCK
    assert blocks > 0, f"{len(ids)} tokens cannot fill one {WIKITEXT2_BLOCK}-token block."
    usable = ids[: blocks * WIKITEXT2_BLOCK + 1]
    return (usable[:-1].view(blocks, WIKITEXT2_BLOCK).clone(),
            usable[1:].view(blocks, WIKITEXT2_BLOCK).clone())


def _tokenize(texts: list[str], checkpoint: str) -> torch.Tensor:
    """Tokenize documents into the [rows, 2, sequence] ids/mask pair the text backbones expect."""
    encoded = AutoTokenizer.from_pretrained(checkpoint)(
        texts, max_length=BERT_SEQUENCE_LENGTH, padding="max_length", truncation=True,
        return_tensors="pt",
    )
    return torch.stack((encoded["input_ids"], encoded["attention_mask"]), dim=1)


# --- Dataset loading --- #
def precompute_dataset_to_shared(
    dataset_name: str,
    data_folder: Path,
    split: str,
    dtype: torch.dtype = torch.float32,
    fraction: float | None = None,
    seed: int | None = None,
    image_size: int = 32,
    max_groups: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Precompute a dataset at the requested resolution and store it in shared memory.

    Returns the features, the labels, and - only for datasets that carry one - a per-sample client
    id (FEMNIST's writer, Sent140's user) for the natural non-IID partition. `None` otherwise.

    `max_groups` is required for those datasets and rejected for the rest. It keeps that many of
    the largest natural groups and applies to the IID split as well as the natural one, so pass
    `num_clients` on both. It is the only place the client pool is chosen: `natural_partition`
    refuses a pool it would have to narrow, so the two arms cannot silently disagree. It also caps
    the memory - FEMNIST's 693,822 images are an 8.5 GiB three-channel float tensor - but the
    reason it is mandatory is the comparison: an IID twin drawn from all 3,057 writers would hold
    forty times the data per client and confound partition with volume.
    """
    dataset_name = dataset_name.upper()
    if dataset_name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_NAMES)}")

    # The group datasets exist to be split one client per natural group, so the caller has to say
    # how many groups it wants and must ask for the same number on both splits and in the IID twin.
    # Left optional, the natural arm silently kept its `num_clients` largest writers while the IID
    # arm drew from all 3,057 - matching the partition but not the data volume, which is the one
    # thing the pair is built to hold constant.
    assert (max_groups is not None) == (dataset_name in GROUP_FIELD), (
        f"max_groups is required for {sorted(GROUP_FIELD)} and meaningless otherwise; got "
        f"max_groups={max_groups} for {dataset_name}."
    )
    is_train = (split == "train")
    data_path = Path(data_folder)
    groups: torch.Tensor | None = None

    # Load dataset
    if dataset_name == 'SYNTHETIC':
        sample_count = 2_000 if is_train else 800
        generator = torch.Generator().manual_seed((seed or 0) + (0 if is_train else 1))
        X = torch.randn(sample_count, 3, 32, 32, generator=generator)
        y = torch.arange(sample_count, dtype=torch.long) % 10
    elif dataset_name == 'SVHN':
        # The only set not downloaded on demand: its .mat files are kept in the repo data folder.
        ds = SVHN(root=data_path / "SVHN", split=split, download=False)
        X = _resize(torch.from_numpy(ds.data).float().div_(255.0), image_size)
        y = torch.tensor(ds.labels, dtype=torch.long)
    elif dataset_name in {'CIFAR10', 'CIFAR100'}:
        dataset_class = CIFAR10 if dataset_name == 'CIFAR10' else CIFAR100
        ds = dataset_class(root=data_path / dataset_name, train=is_train, download=True)
        X = _resize(torch.from_numpy(ds.data).float().permute(0, 3, 1, 2).div_(255.0), image_size)
        y = torch.tensor(ds.targets, dtype=torch.long)
    elif dataset_name == 'STL10':
        ds = STL10(root=data_path / "STL10", split=split, download=True)
        X = _resize(torch.from_numpy(ds.data).float().div_(255.0), image_size)
        y = torch.tensor(ds.labels, dtype=torch.long)
    elif dataset_name == 'MNIST':
        ds = MNIST(root=data_path / "MNIST", train=is_train, download=True)
        # Resize on the single channel, then repeat it, so the three-channel backbones need no
        # special case and the resample runs over a third of the data.
        X = _resize(ds.data.unsqueeze(1).float().div_(255.0), image_size).repeat(1, 3, 1, 1)
        y = ds.targets.clone().to(torch.long)
    elif dataset_name == 'IMAGENETTE':
        # Imagenette refuses to download over an existing copy, so ask only when it is absent.
        root = data_path / "IMAGENETTE"
        ds = Imagenette(
            root=root, split="train" if is_train else "val", size="160px",
            download=not (root / "imagenette2-160").exists(),
        )
        X, y = _decode_images(ds, list(range(len(ds))), image_size)
    elif dataset_name == 'EUROSAT':
        ds = EuroSAT(root=data_path / "EUROSAT", download=True)
        order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(SPLIT_SEED))
        cut = int(len(ds) * EUROSAT_TRAIN_FRACTION)
        X, y = _decode_images(ds, (order[:cut] if is_train else order[cut:]).tolist(), image_size)
    elif dataset_name == 'FOOD101':
        ds = Food101(root=data_path / "FOOD101", split="train" if is_train else "test",
                     download=True)
        X, y = _decode_images(ds, list(range(len(ds))), image_size)
    elif dataset_name == 'FEMNIST':
        from datasets import load_dataset

        # One split upstream, so a deterministic 85/15 cut by *writer* keeps every writer wholly on
        # one side - otherwise the natural partition would leak across train and test.
        source = load_dataset("flwrlabs/femnist", split="train", cache_dir=str(data_path / "hf"))
        writers = np.array(source["writer_id"])
        unique = np.unique(writers)
        order = np.random.default_rng(SPLIT_SEED).permutation(len(unique))
        keep = set(unique[order[: int(0.85 * len(unique))]].tolist())
        rows = [i for i, w in enumerate(writers) if (w in keep) == is_train]
        # Capped before decoding, not after: the whole split is 8.5 GiB once it is three-channel
        # float, which is more memory than this machine has to spare.
        rows = _largest_groups(rows, writers[rows], max_groups)
        images = torch.stack([transforms.PILToTensor()(source[i]["image"].convert("L")) for i in rows])
        X = _resize(images.float().div_(255.0), image_size).repeat(1, 3, 1, 1)
        y = torch.tensor([source[i]["character"] for i in rows], dtype=torch.long)
        groups = _group_ids(writers[rows])
    elif dataset_name == 'TINYIMAGENET':
        from datasets import load_dataset

        # Native 64x64, so `_resize` is a no-op at the setup's own resolution. A handful of images
        # are greyscale, hence the explicit RGB conversion.
        source = load_dataset(TINYIMAGENET_REPO, split="train" if is_train else "valid",
                              cache_dir=str(data_path / "hf"))
        images = torch.stack([
            transforms.PILToTensor()(record["image"].convert("RGB")) for record in source
        ])
        X = _resize(images.float().div_(255.0), image_size)
        y = torch.tensor(source["label"], dtype=torch.long)
    elif dataset_name == 'WIKITEXT2':
        X, y = _wikitext_blocks(data_path, is_train)
    elif dataset_name == 'DBPEDIA':
        from datasets import load_dataset

        source = load_dataset(DBPEDIA_REPO, split="train" if is_train else "test",
                              cache_dir=str(data_path / "hf"))
        if is_train:
            source = source.shuffle(seed=SPLIT_SEED).select(range(DBPEDIA_SAMPLES))
        # Title and content both carry the ontology signal, and the title alone is often decisive.
        X = _tokenize([f"{t}. {c}" for t, c in zip(source["title"], source["content"])],
                      TEXT_CHECKPOINT[dataset_name])
        y = torch.tensor(source["label"], dtype=torch.long)
    elif dataset_name == 'YELP5':
        from datasets import load_dataset

        source = load_dataset(YELP5_REPO, split="train" if is_train else "test",
                              cache_dir=str(data_path / "hf"))
        if is_train:
            source = source.shuffle(seed=SPLIT_SEED).select(range(YELP5_SAMPLES))
        X = _tokenize(list(source["text"]), TEXT_CHECKPOINT[dataset_name])
        y = torch.tensor(source["label"], dtype=torch.long)
    elif dataset_name == 'AGNEWS':
        from datasets import load_dataset

        source = load_dataset("ag_news", split="train" if is_train else "test",
                              cache_dir=str(data_path / "hf"))
        if is_train:
            source = source.shuffle(seed=SPLIT_SEED).select(range(AGNEWS_SAMPLES))
        X = _tokenize(list(source["text"]), TEXT_CHECKPOINT[dataset_name])
        y = torch.tensor(source["label"], dtype=torch.long)
    elif dataset_name == 'SENT140':
        from datasets import load_dataset

        frame = load_dataset(
            SENT140_REPO, split="train", cache_dir=str(data_path / "hf")
        ).to_pandas().dropna(subset=["text", "user", "sentiment"])
        # Kept by *user*, not by row: sampling rows would flatten the natural partition before it
        # is ever applied. Busiest users first so a one-user client still has data to train on.
        busiest = frame["user"].value_counts()
        keep = list(busiest.index[: SENT140_USERS])
        cut = int(0.85 * len(keep))
        wanted = set(keep[:cut]) if is_train else set(keep[cut:])
        ranked = list(busiest.index[busiest.index.isin(wanted)])
        assert len(ranked) >= max_groups, (
            f"{len(ranked)} Sent140 users in the {split} split cannot supply {max_groups} clients.")
        wanted = set(ranked[:max_groups])
        frame = frame[frame["user"].isin(wanted)].reset_index(drop=True)
        X = _tokenize(frame["text"].astype(str).tolist(), TEXT_CHECKPOINT[dataset_name])
        # Positive is 4 in the original encoding and 1 in some mirrors; treat anything above zero
        # as positive so a mirror's convention cannot silently collapse this to one class.
        y = torch.tensor((frame["sentiment"].to_numpy() > 0).astype("int64"), dtype=torch.long)
        assert y.unique().numel() == 2, (
            f"SENT140 {split} split collapsed to one class; sentiment values were "
            f"{sorted(set(frame['sentiment'].unique().tolist()))[:5]}."
        )
        groups = _group_ids(frame["user"].to_numpy())
    else:  # 20NEWS
        ds = fetch_20newsgroups(
            data_home=data_path / "20NEWS",
            subset=split,
            shuffle=False,
            remove=('headers', 'footers', 'quotes'),
        )
        X = _tokenize(list(ds.data), TEXT_CHECKPOINT[dataset_name])
        y = torch.tensor(ds.target, dtype=torch.long)

    # Train is always shuffled; test only needs shuffling when a fraction is
    # taken, so the subset is not biased by the original file order.
    if is_train or fraction is not None:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        perm = torch.randperm(len(y), generator=generator)
        X, y = X[perm], y[perm]
        if groups is not None:
            groups = groups[perm]
        if fraction is not None:
            # Clone so share_memory_ ships only the subset, not the full-size storage
            # the [:n] view would otherwise keep alive.
            n = int(len(y) * fraction)
            X, y = X[:n].clone(), y[:n].clone()
            if groups is not None:
                groups = groups[:n].clone()

    if dataset_name in DATASET_CONFIG:
        # Normalize in place to avoid transient full-size copies of image datasets.
        mean_values, std_values = DATASET_CONFIG[dataset_name]
        mean = torch.tensor(mean_values).view(1, 3, 1, 1)
        std = torch.tensor(std_values).view(1, 3, 1, 1)
        X = X.sub_(mean).div_(std).to(dtype)

    X.share_memory_()
    y.share_memory_()
    if groups is not None:
        groups.share_memory_()
    return X, y, groups


# --- Client partitioning --- #
def natural_partition(groups: torch.Tensor, num_clients: int, seed: int) -> list[list[int]]:
    """Give each client its own natural group: one FEMNIST writer, one Sent140 user.

    The pool must already hold exactly one group per client - `_largest_groups`, at load time, is
    what chooses it, so the natural arm and its IID twin are provably drawn from the same writers.
    Selecting the pool here instead would let the two arms disagree silently.

    Two limits worth knowing before reading a result off this. The pool is the largest groups, so a
    small `num_clients` is an experiment on the most prolific writers, not on FEMNIST. And a writer
    split moves handwriting style, not the label distribution - FEMNIST writers each cover
    essentially the whole 62-character set, so this is close to label-IID and does not substitute
    for `dirichlet_partition`.
    """
    members: dict[int, list[int]] = {}
    for index, group in enumerate(groups.tolist()):
        members.setdefault(group, []).append(index)
    assert len(members) == num_clients, (
        f"The natural pool holds {len(members)} groups for {num_clients} clients; load the data "
        f"with max_groups={num_clients} so this split and its IID twin share one pool."
    )

    order = torch.randperm(num_clients, generator=torch.Generator().manual_seed(seed)).tolist()
    pool = list(members.values())
    buckets = [pool[pick] for pick in order]
    assert all(buckets), "a client received no samples from its natural group"
    return buckets


def dirichlet_partition(
    labels: torch.Tensor, num_clients: int, seed: int, alpha: float
) -> list[list[int]]:
    """Label-skewed non-IID split, for the datasets that carry no client ids of their own.

    The standard Dirichlet benchmark: each client draws a class mixture from Dir(alpha) and fills
    its quota from those classes, so a small alpha leaves a client holding only a few classes. Used
    for Imagenette and EuroSAT, which unlike FEMNIST and Sent140 have no real clients to partition
    by - see `natural_partition` for those.

    Every client is given the *same* number of samples, which the usual formulation does not do.
    That is deliberate: the IID twin of these setups holds len(dataset)/num_clients each, and a
    split that changed volume as well as label mix would confound the one comparison the pair exists
    to make. This is the mistake that had to be corrected on FEMNIST.
    """
    assert alpha > 0, f"Dirichlet alpha must be positive; got {alpha}."
    # A sequence target has one label per position, so `nonzero` would return [row, column] pairs
    # and flattening them mixes column numbers into the sample pools. The result is a partition of
    # the right size that silently repeats and overlaps rows, which nothing downstream detects.
    assert labels.ndim == 1, (
        f"Dirichlet label skew needs one label per sample; got targets of shape "
        f"{tuple(labels.shape)}. A sequence target has a vocabulary, not a label set."
    )
    generator = np.random.default_rng(seed)
    classes = labels.unique().tolist()
    pools = {
        label: generator.permutation(torch.nonzero(labels == label).flatten().numpy()).tolist()
        for label in classes
    }
    quota = len(labels) // num_clients
    assert quota, f"{len(labels)} samples cannot fill {num_clients} clients."

    buckets: list[list[int]] = []
    for _ in range(num_clients):
        mixture = generator.dirichlet([alpha] * len(classes))
        taken: list[int] = []
        for label, count in zip(classes, generator.multinomial(quota, mixture)):
            taken.extend(pools[label][:count])
            del pools[label][:count]
        # A class can run dry before its share is filled, and the last clients drain what is left.
        # Top up from the largest remaining pool so every client still ends on the same quota.
        while len(taken) < quota:
            biggest = max(pools, key=lambda label: len(pools[label]))
            assert pools[biggest], "ran out of samples before every client was filled"
            need = quota - len(taken)
            taken.extend(pools[biggest][:need])
            del pools[biggest][:need]
        buckets.append(taken)
    return buckets


def client_partitions(
    y: torch.Tensor, groups: torch.Tensor | None, cfg: Any, num_clients: int
) -> list[list[int]]:
    """Return the sample indices each client trains on under the configured split.

    The single source of truth for who holds what: `create_dataloader` builds its subset from this,
    and the pre-spawn class-coverage checks validate this rather than assuming the IID stride.

    Branching on the *request*, not on whether group ids happen to be present: keying off
    `groups is not None` meant asking for a non-IID split on a dataset without client ids silently
    returned the IID stride instead of saying it could not comply.
    """
    if not cfg.non_iid_partition:
        perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(cfg.seed))
        return [perm[client_id::num_clients].tolist() for client_id in range(num_clients)]
    if groups is not None:
        return natural_partition(groups, num_clients, cfg.seed)
    return dirichlet_partition(y, num_clients, cfg.seed, cfg.dirichlet_alpha)


def create_dataloader(
    X: torch.Tensor, y: torch.Tensor, cfg: Any, device: torch.device,
    is_train: bool, client_id: int | None = None, num_clients: int | None = None,
    groups: torch.Tensor | None = None
) -> DataLoader:
    """Create a train or test dataloader over shared tensors."""
    assert (client_id is None and is_train is False) or (client_id is not None and is_train is True)

    dataset = SharedTensorDataset(X, y)

    if is_train:
        assert client_id is not None and num_clients is not None, "client_id and num_clients required for training"
        dataset = Subset(dataset, client_partitions(y, groups, cfg, num_clients)[client_id])
        shuffle_generator = torch.Generator().manual_seed(cfg.seed + client_id + 1)

        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            pin_memory=(device.type == "cuda"),
            num_workers=cfg.num_loader_workers,
            prefetch_factor=2 if cfg.num_loader_workers > 0 else None,
            persistent_workers=cfg.num_loader_workers > 0,
            shuffle=True,
            generator=shuffle_generator,
        )
    else:
        # Testing: use full dataset, no shuffle, larger batch
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
            num_workers=0
        )


# --- Debug-only split repair --- #
def force_class_coverage(y_train: torch.Tensor, y_test: torch.Tensor, cfg: FLConfig) -> None:
    """Brute-force label flips so fractioned debug splits still cover every class.

    Subsetting via dataset_fraction can drop classes from the test split or from a
    client's train partition, which breaks the AUC metrics downstream. This flips a
    few labels of the most frequent class to the missing ones — deliberately wrong
    labels, acceptable only for fraction-reduced debug runs.

    Sequence targets have a vocabulary rather than a label set and are left alone. So are the
    client partitions of a non-IID run, where an incomplete label set is the point of the split;
    `evaluate` reports AUC as NaN for those clients instead. Only the IID branch is reached, so no
    group ids are needed to rebuild the partition.
    """
    assert cfg.num_classes is not None
    if y_train.ndim > 1:
        return
    expected_classes = set(range(cfg.num_classes))

    def flip_missing(y: torch.Tensor, positions: torch.Tensor, split_name: str) -> None:
        labels = y[positions]
        missing = sorted(expected_classes - set(torch.unique(labels).tolist()))
        if not missing:
            return
        donor_class = int(torch.bincount(labels, minlength=cfg.num_classes).argmax())
        donor_positions = positions[labels == donor_class]
        assert len(donor_positions) > len(missing), (
            f"Cannot repair class coverage for {split_name}: {len(donor_positions)} samples of "
            f"donor class {donor_class} cannot cover {len(missing)} missing classes."
        )
        for donor_pos, missing_class in zip(donor_positions.tolist(), missing):
            y[donor_pos] = missing_class
        print(f"[Debug] Flipped {len(missing)} label(s) in {split_name} to cover missing classes {missing}.")

    flip_missing(y_test, torch.arange(len(y_test)), "test split")
    if cfg.non_iid_partition:
        return
    for client_id, indices in enumerate(client_partitions(y_train, None, cfg, cfg.num_clients)):
        flip_missing(y_train, torch.tensor(indices), f"client {client_id} train partition")
