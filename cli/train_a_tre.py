"""Train a composition head on top of a frozen audio encoder for A-TRE.

The encoder caches for the train + val splits are built on first run and
reused thereafter. The trained head is saved at ``<output.dir>/tre_head.pth``.

Usage::

    python cli/train_a_tre.py configs/random.yaml
    python cli/train_a_tre.py configs/random.yaml train.epochs=5 train.batch_size=128
"""
from __future__ import annotations

from audiocomp.cli_utils import get_view, parse_args
from audiocomp.data import ATREDataset
from audiocomp.models import get_model
from audiocomp.tasks import train_a_tre


def main() -> None:
    config_path, cfg, device = parse_args(__doc__)
    print(f"Model : {cfg['model']['name']}")
    print(f"Config: {config_path}")
    print(f"Device: {device}")

    encoder = get_model(**{k: v for k, v in cfg["model"].items() if v is not None})

    train_ds = ATREDataset(split="train", target_sr=encoder.sample_rate)
    val_ds = ATREDataset(split="val", target_sr=encoder.sample_rate)
    train_view = get_view(
        encoder, cfg, train_ds, "a-tre-10k", "train",
        embedding_shape=(encoder.embedding_dim,),
        device=device,
    )
    val_view = get_view(
        encoder, cfg, val_ds, "a-tre-10k", "val",
        embedding_shape=(encoder.embedding_dim,),
        device=device,
    )

    t = cfg["train"]
    ckpt_path = train_a_tre(
        train_view, val_view, encoder.embedding_dim,
        output_dir=cfg["output"]["dir"],
        epochs=t["epochs"],
        batch_size=t["batch_size"],
        lr=t["lr"],
        weight_decay=t["weight_decay"],
        grad_clip=t["grad_clip"],
        patience=t["patience"],
        ema_alpha=t["ema_alpha"],
        seed=t["seed"],
        device=device,
    )
    print(f"\nDone. Best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
