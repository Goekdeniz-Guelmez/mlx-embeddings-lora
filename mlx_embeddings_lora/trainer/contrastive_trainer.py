import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map
from mlx_lm.tuner.trainer import grad_checkpoint
from tqdm import tqdm

from .dataset import CacheDataset


@dataclass
class ContrastiveTrainingArgs:
    batch_size: int = field(default=4, metadata={"help": "Minibatch size."})
    iters: int = field(default=100, metadata={"help": "Iterations to train for."})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of gradient accumulation steps."}
    )
    val_batches: int = field(
        default=25,
        metadata={
            "help": "Number of validation batches, -1 uses the entire validation set."
        },
    )
    steps_per_report: int = field(
        default=10,
        metadata={"help": "Number of training steps between loss reporting."},
    )
    steps_per_eval: int = field(
        default=200, metadata={"help": "Number of training steps between validations."}
    )
    steps_per_save: int = field(
        default=100, metadata={"help": "Save the model every number steps"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length."}
    )
    adapter_file: str = field(
        default="adapters.safetensors",
        metadata={"help": "Save/load path for the trained adapter weights."},
    )
    grad_checkpoint: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to reduce memory use."},
    )
    temperature: float = field(
        default=0.07,
        metadata={"help": "For infonce and nt_xent."},
    )
    margin: float = field(
        default=0.5,
        metadata={"help": "For triplet loss."},
    )
    similarity: str = field(
        default="cosine",
        metadata={"help": "Similarity calculation: sine and cosine."},
    )


def sine_similarity(x: mx.array, y: mx.array) -> mx.array:
    cos_sim = mx.sum(x * y, axis=-1)
    return mx.sqrt(mx.maximum(0, 1 - cos_sim**2))


def cosine_similarity(x: mx.array, y: mx.array) -> mx.array:
    return mx.sum(x * y, axis=-1)


def triplet_loss(
    anchor: mx.array,
    positive: mx.array,
    negative: mx.array,
    margin: float = 0.0,
    similarity: str = "cosine",
):
    """
    Triplet Loss: max(0, d(a,p) - d(a,n) + margin)
    Using distance = 1 - cosine_similarity
    """
    if similarity == "cosine":
        pos_sim = cosine_similarity(anchor, positive)
        neg_sim = cosine_similarity(anchor, negative)
    elif similarity == "sine":
        pos_sim = sine_similarity(anchor, positive)
        neg_sim = sine_similarity(anchor, negative)
    else:
        pos_sim = cosine_similarity(anchor, positive)
        neg_sim = cosine_similarity(anchor, negative)

    # Distance = 1 - similarity (since embeddings are normalized)
    pos_dist = 1 - pos_sim
    neg_dist = 1 - neg_sim

    loss = mx.maximum(0, pos_dist - neg_dist + margin)
    return loss


def infonce_loss(
    anchor: mx.array,
    positive: mx.array,
    negative: mx.array,
    temperature: float = 1.0,
    similarity: str = "cosine",
):
    """
    InfoNCE Loss with single negative
    """
    if similarity == "cosine":
        pos_sim = cosine_similarity(anchor, positive) / temperature
        neg_sim = cosine_similarity(anchor, negative) / temperature
    elif similarity == "sine":
        pos_sim = sine_similarity(anchor, positive) / temperature
        neg_sim = sine_similarity(anchor, negative) / temperature
    else:
        pos_sim = cosine_similarity(anchor, positive) / temperature
        neg_sim = cosine_similarity(anchor, negative) / temperature

    logits = mx.stack([pos_sim, neg_sim], axis=-1)
    return -pos_sim + mx.logsumexp(logits, axis=-1)


def multiple_negatives_ranking_loss(
    anchor: mx.array, positive: mx.array, negative: mx.array, similarity: str = "cosine"
):
    """
    Per-sample Multiple Negatives Ranking Loss
    """
    if similarity == "cosine":
        pos_sim = cosine_similarity(anchor, positive)
        neg_sim = cosine_similarity(anchor, negative)
    elif similarity == "sine":
        pos_sim = -sine_similarity(anchor, positive)
        neg_sim = -sine_similarity(anchor, negative)
    else:
        pos_sim = cosine_similarity(anchor, positive)
        neg_sim = cosine_similarity(anchor, negative)

    if len(neg_sim.shape) == 1:
        logits = mx.stack([pos_sim, neg_sim], axis=-1)
        labels = mx.zeros(anchor.shape[0], dtype=mx.int32)
    else:
        logits = mx.concatenate([pos_sim[:, None], neg_sim], axis=-1)
        labels = mx.zeros(anchor.shape[0], dtype=mx.int32)

    losses = nn.losses.cross_entropy(logits, labels, reduction="none")
    return mx.mean(losses)


def nt_xent_loss(
    anchor: mx.array,
    positive: mx.array,
    negative: mx.array,
    temperature: float = 1.0,
    similarity: str = "cosine",
):
    """
    NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)
    Used in SimCLR - similar to InfoNCE but with softmax over all negatives
    """
    if similarity == "cosine":
        pos_sim = cosine_similarity(anchor, positive) / temperature
        neg_sim = cosine_similarity(anchor, negative) / temperature
    elif similarity == "sine":
        pos_sim = -sine_similarity(anchor, positive) / temperature
        neg_sim = -sine_similarity(anchor, negative) / temperature
    else:
        pos_sim = cosine_similarity(anchor, positive) / temperature
        neg_sim = cosine_similarity(anchor, negative) / temperature

    # Stack similarities: [positive_sim, negative_sims...]
    if len(neg_sim.shape) == 1:
        neg_sim = neg_sim[None, :]  # Add dimension if single negative
    logits = mx.concatenate([pos_sim[..., None], neg_sim], axis=-1)

    # Apply log softmax and take negative log prob of positive (index 0)
    log_softmax = nn.log_softmax(logits, axis=-1)
    loss = -log_softmax[..., 0]
    return loss


def iterate_batches(dataset, batch_size, max_seq_length, train=False):
    if isinstance(dataset, CacheDataset):
        len_fn = lambda idx: dataset.itemlen(idx)
    else:
        len_fn = lambda idx: len(dataset[idx][0])  # use anchor length

    idx = sorted(range(len(dataset)), key=len_fn)
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size}"
            f" examples but only has {len(dataset)}."
        )

    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    batch_idx = [
        idx[i : i + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]

    while True:
        indices = np.random.permutation(len(batch_idx))
        for i in indices:
            batch_triplets = [dataset[j] for j in batch_idx[i]]
            anchors, positives, negatives = zip(*batch_triplets)

            def pad_batch(sequences):
                lengths = [len(x) for x in sequences]
                pad_to = 32
                max_length = 1 + pad_to * ((max(lengths) + pad_to - 1) // pad_to)
                max_length = min(max_length, max_seq_length)

                batch_arr = np.zeros((batch_size // step, max_length), np.int32)
                for j in range(batch_size // step):
                    truncated = min(lengths[j], max_seq_length)
                    batch_arr[j, :truncated] = sequences[j][:truncated]
                    lengths[j] = truncated
                return mx.array(batch_arr), mx.array(lengths)

            anchors, anchor_lens = pad_batch(anchors)
            positives, pos_lens = pad_batch(positives)
            negatives = (
                pad_batch(negatives)[0]
                if negatives[0] is not None
                else None  # can be None → use in-batch negatives
            )

            yield anchors, positives, negatives, anchor_lens, pos_lens

        if not train:
            break


def map_loss_functoins(
    loss_fn: Optional[callable],
    margin: Optional[float] = 0.0,
    temperature: Optional[float] = 1.0,
    similarity: Optional[str] = "cosine",
):
    loss_functions = {
        "triplet": lambda a, p, n: triplet_loss(
            a, p, n, margin=margin, similarity=similarity
        ),
        "infonce": lambda a, p, n: infonce_loss(
            a, p, n, temperature=temperature, similarity=similarity
        ),
        "mnr": lambda a, p, n: multiple_negatives_ranking_loss(
            a, p, n, similarity=similarity
        ),
        "nt_xent": lambda a, p, n: nt_xent_loss(
            a, p, n, temperature=temperature, similarity=similarity
        ),
    }
    return loss_functions.get(loss_fn, loss_functions["infonce"])


def create_in_batch_negatives(anchor_emb: mx.array, positive_emb: mx.array) -> mx.array:
    batch_size = anchor_emb.shape[0]
    if batch_size == 1:
        return mx.random.normal(positive_emb.shape)
    indices = mx.concatenate([mx.arange(1, batch_size), mx.array([0])])
    negative_emb = anchor_emb[indices]
    return negative_emb


def evaluate_contrastive(
    model,
    dataset,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss_fn: str = "infonce",
    similarity: str = "cosine",
    temperature: float = 0.0,
    margin: float = 0.0,
    iterate_batches: callable = iterate_batches,
):
    model.eval()
    all_losses = []
    loss_func = map_loss_functoins(
        loss_fn=loss_fn, margin=margin, temperature=temperature, similarity=similarity
    )
    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)
    for _, batch in zip(
        index_iterator,
        iterate_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        anchors, positives, negatives, anchor_lens, pos_lens = batch
        anchor_output = model(anchors)
        positive_output = model(positives)

        anchor_emb = anchor_output.text_embeds
        positive_emb = positive_output.text_embeds

        if negatives is not None:
            negative_output = model(negatives)
            negative_emb = negative_output.text_embeds
        else:
            negative_emb = create_in_batch_negatives(anchor_emb, positive_emb)

        losses = loss_func(anchor_emb, positive_emb, negative_emb)
        all_losses.append(losses)
        mx.eval(losses)
    all_losses = mx.concatenate(all_losses) if all_losses else mx.array([0.0])
    avg_loss = mx.mean(all_losses)
    avg_loss = mx.distributed.all_sum(avg_loss) / mx.distributed.init().size()
    return avg_loss.item()


def train_contrastive(
    model,
    optimizer,
    train_dataset,
    val_dataset,
    args: ContrastiveTrainingArgs = ContrastiveTrainingArgs(),
    loss_fn: str = "infonce",
    similarity: str = "cosine",
    iterate_batches: callable = iterate_batches,
    training_callback=None,
):
    mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
    tqdm.write(
        f"Starting embedding training with {loss_fn} loss fn, iters: {args.iters}"
    )
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        tqdm.write(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.model.layers[0])

    grad_accum_steps = args.gradient_accumulation_steps
    if grad_accum_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")

    loss_func = map_loss_functoins(
        loss_fn=loss_fn,
        margin=args.margin,
        temperature=args.temperature,
        similarity=similarity,
    )

    def embedding_loss(model, anchors, positives, negatives, anchor_lens, pos_lens):
        anchor_output = model(anchors)
        positive_output = model(positives)

        anchor_emb = anchor_output.text_embeds
        positive_emb = positive_output.text_embeds

        if negatives is not None:
            negative_output = model(negatives)
            negative_emb = negative_output.text_embeds
        else:
            negative_emb = create_in_batch_negatives(anchor_emb, positive_emb)

        losses = loss_func(anchor_emb, positive_emb, negative_emb)
        return mx.mean(losses), anchor_lens.sum()

    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, toks, grad

    loss_value_and_grad = nn.value_and_grad(model, embedding_loss)

    model.train()
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0
    grad_accum = None

    # Main training loop
    pbar = tqdm(range(1, args.iters + 1), desc="Training", disable=rank != 0)
    for it in pbar:
        batch = next(
            iterate_batches(
                dataset=train_dataset,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                train=True,
            )
        )

        if args.steps_per_eval is not None and (
            it == 1 or it % args.steps_per_eval == 0 or it == args.iters
        ):
            tic = time.perf_counter()
            val_loss = evaluate(
                model=model,
                dataset=val_dataset,
                loss_fn=loss_fn,
                similarity=similarity,
                temperature=args.temperature,
                margin=args.margin,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                iterate_batches=iterate_batches,
            )
            model.train()
            val_time = time.perf_counter() - tic
            if rank == 0:
                tqdm.write(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {val_time:.3f}s",
                )

            if training_callback is not None:
                val_info = {
                    "iteration": it,
                    "val_loss": val_loss,
                    "val_time": val_time,
                }
                training_callback.on_val_loss_report(val_info)

        tic = time.perf_counter()

        lvalue, toks, grad_accum = step(
            batch,
            grad_accum,
            it % grad_accum_steps == 0,
        )
        losses += lvalue
        n_tokens += toks
        steps += 1
        mx.eval(state, losses, n_tokens, grad_accum)
        train_time += time.perf_counter() - tic

        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * world_size
            n_tokens = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / train_time
            tokens_sec = float(n_tokens) / train_time
            trained_tokens += n_tokens
            peak_mem = mx.get_peak_memory() / 1e9
            if rank == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{train_loss:.3f}",
                        "it/s": f"{it_sec:.3f}",
                    }
                )
                tqdm.write(
                    f"\nIter {it}: "
                    f"loss {train_loss:.3f}, "
                    f"lr {learning_rate:.3e}, "
                    f"it/s {it_sec:.3f}, "
                    f"tok/s {tokens_sec:.3f}, "
                    f"trained_tok {trained_tokens}, "
                    f"peak_mem {peak_mem:.3f}GB"
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            n_tokens = 0
            steps = 0
            train_time = 0

        if it % args.steps_per_save == 0 and rank == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            tqdm.write(
                f"\n"
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    if rank == 0:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(args.adapter_file), adapter_weights)
        tqdm.write(f"Saved final weights to {args.adapter_file}.")
