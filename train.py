import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import (
    CherenkovDataset,
    collate_cherenkov_generation,
)

from model import (
    CherenkovGenerator,
    GeneratorConfig,
)


# ================================================================
# CONFIGURATION
# ================================================================

H5_PATH = "/content/drive/MyDrive/tokenized_training.h5"

CHECKPOINT_DIR = "checkpoints"

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "cherenkov_foundation_best.pt"
)

LAST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "cherenkov_foundation_last.pt"
)


# ------------------------------------------------
# Training
# ------------------------------------------------

BATCH_SIZE = 64

NUM_EPOCHS = 5

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

TRAIN_FRACTION = 0.90

RANDOM_SEED = 42


# ------------------------------------------------
# DataLoader
# ------------------------------------------------

NUM_WORKERS = 0

PIN_MEMORY = True


# ------------------------------------------------
# Loss
# ------------------------------------------------

# Empty slots are extremely common because
# N_MAX_HITS = 192 while the average event has
# only ~52 photons.
#
# Therefore we don't want the occupancy loss
# to completely dominate the training.
EMPTY_CLASS_WEIGHT = 0.10


# Relative weights of the different losses

OCCUPANCY_LOSS_WEIGHT = 1.0

X_LOSS_WEIGHT = 1.0

Y_LOSS_WEIGHT = 1.0

ENERGY_LOSS_WEIGHT = 1.0


# ================================================================
# REPRODUCIBILITY
# ================================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# ================================================================
# DEVICE
# ================================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("Using CUDA.")

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        device = torch.device("cpu")

        print("CUDA unavailable.")

        print("Using CPU.")

    return device


# ================================================================
# LOSS
# ================================================================

class CherenkovLoss(nn.Module):
    """
    Loss for the non-autoregressive Cherenkov generator.

    Components:

        1. Occupancy
        2. x coordinate
        3. y coordinate
        4. photon energy

    x/y/energy losses are calculated ONLY on positions
    containing a real photon.

    Empty slots therefore do not contribute to the
    coordinate/energy losses.
    """

    def __init__(self, config):

        super().__init__()

        # --------------------------------------------------------
        # Occupancy loss
        # --------------------------------------------------------

        # Class 0 = EMPTY
        # Class 1 = HIT

        occupancy_weights = torch.tensor(
            [
                EMPTY_CLASS_WEIGHT,
                1.0,
            ],
            dtype=torch.float32
        )

        self.register_buffer(
            "occupancy_weights",
            occupancy_weights
        )

        self.x_loss = nn.CrossEntropyLoss(
            reduction="none"
        )

        self.y_loss = nn.CrossEntropyLoss(
            reduction="none"
        )

        self.energy_loss = nn.CrossEntropyLoss(
            reduction="none"
        )

    def forward(
        self,
        outputs,
        batch
    ):

        occupancy_logits = outputs[
            "occupancy_logits"
        ]

        x_logits = outputs[
            "x_logits"
        ]

        y_logits = outputs[
            "y_logits"
        ]

        energy_logits = outputs[
            "energy_logits"
        ]

        occupancy_target = batch[
            "occupancy"
        ]

        x_target = batch[
            "x_tokens"
        ]

        y_target = batch[
            "y_tokens"
        ]

        energy_target = batch[
            "energy_tokens"
        ]

        # --------------------------------------------------------
        # Occupancy
        # --------------------------------------------------------

        occupancy_loss = nn.functional.cross_entropy(
            occupancy_logits.transpose(1, 2),
            occupancy_target,
            weight=self.occupancy_weights
        )

        # --------------------------------------------------------
        # HIT mask
        # --------------------------------------------------------

        hit_mask = occupancy_target == 1

        number_of_hits = hit_mask.sum()

        # --------------------------------------------------------
        # x/y/energy losses
        # --------------------------------------------------------

        if number_of_hits > 0:

            x_loss = self.x_loss(
                x_logits[hit_mask],
                x_target[hit_mask]
            ).mean()

            y_loss = self.y_loss(
                y_logits[hit_mask],
                y_target[hit_mask]
            ).mean()

            energy_loss = self.energy_loss(
                energy_logits[hit_mask],
                energy_target[hit_mask]
            ).mean()

        else:

            # This is extremely unlikely for a whole batch,
            # but protects against division by zero.

            x_loss = torch.zeros(
                (),
                device=occupancy_logits.device
            )

            y_loss = torch.zeros(
                (),
                device=occupancy_logits.device
            )

            energy_loss = torch.zeros(
                (),
                device=occupancy_logits.device
            )

        # --------------------------------------------------------
        # Total
        # --------------------------------------------------------

        total_loss = (
            OCCUPANCY_LOSS_WEIGHT * occupancy_loss
            +
            X_LOSS_WEIGHT * x_loss
            +
            Y_LOSS_WEIGHT * y_loss
            +
            ENERGY_LOSS_WEIGHT * energy_loss
        )

        return {
            "total": total_loss,
            "occupancy": occupancy_loss,
            "x": x_loss,
            "y": y_loss,
            "energy": energy_loss,
        }


# ================================================================
# EXPERT STATUS
# ================================================================

def print_expert_status(model):

    print()
    print("=" * 60)
    print("EXPERT STATUS")
    print("=" * 60)

    experts = [
        (
            "Expert 0 (refractive index)",
            model.layers[0].moe.n_expert
        ),
        (
            "Expert 1 (momentum)",
            model.layers[0].moe.momentum_expert
        ),
        (
            "Expert 2 (reserved)",
            model.layers[0].moe.reserved_expert_2
        ),
        (
            "Expert 3 (reserved)",
            model.layers[0].moe.reserved_expert_3
        ),
    ]

    for name, expert in experts:

        trainable = any(
            p.requires_grad
            for p in expert.parameters()
        )

        parameters = sum(
            p.numel()
            for p in expert.parameters()
        )

        print(
            f"{name:32s} "
            f"trainable={str(trainable):5s} "
            f"parameters={parameters:,}"
        )


# ================================================================
# PARAMETER COUNTS
# ================================================================

def print_parameter_count(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen = total - trainable

    print()
    print("=" * 60)
    print("MODEL PARAMETERS")
    print("=" * 60)

    print(
        f"Total parameters:     {total:,}"
    )

    print(
        f"Trainable parameters: {trainable:,}"
    )

    print(
        f"Frozen parameters:    {frozen:,}"
    )


# ================================================================
# TRAIN ONE EPOCH
# ================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch
):

    model.train()

    running = {
        "total": 0.0,
        "occupancy": 0.0,
        "x": 0.0,
        "y": 0.0,
        "energy": 0.0,
    }

    number_of_batches = 0

    for batch_index, batch in enumerate(loader):

        # --------------------------------------------------------
        # Move conditioning variables to device
        # --------------------------------------------------------

        momentum = batch[
            "momentum"
        ].to(device, non_blocking=True)

        refractive_index = batch[
            "refractive_index"
        ].to(device, non_blocking=True)

        # --------------------------------------------------------
        # Targets
        # --------------------------------------------------------

        batch["occupancy"] = batch[
            "occupancy"
        ].to(device, non_blocking=True)

        batch["x_tokens"] = batch[
            "x_tokens"
        ].to(device, non_blocking=True)

        batch["y_tokens"] = batch[
            "y_tokens"
        ].to(device, non_blocking=True)

        batch["energy_tokens"] = batch[
            "energy_tokens"
        ].to(device, non_blocking=True)

        # --------------------------------------------------------
        # Gradient reset
        # --------------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # --------------------------------------------------------
        # Forward
        # --------------------------------------------------------

        use_amp = device.type == "cuda"

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):

            outputs = model(
                momentum=momentum,
                refractive_index=refractive_index,
            )

            losses = criterion(
                outputs,
                batch
            )

        # --------------------------------------------------------
        # Backpropagation
        # --------------------------------------------------------

        if use_amp:

            scaler.scale(
                losses["total"]
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            losses["total"].backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

        # --------------------------------------------------------
        # Accumulate
        # --------------------------------------------------------

        for key in running:

            running[key] += (
                losses[key]
                .detach()
                .item()
            )

        number_of_batches += 1

        # --------------------------------------------------------
        # Progress
        # --------------------------------------------------------

        if (
            batch_index + 1
        ) % 100 == 0:

            print(
                f"Epoch {epoch:03d} | "
                f"Batch {batch_index + 1:05d}/{len(loader):05d} | "
                f"Loss {losses['total'].item():.4f}"
            )

    # ------------------------------------------------------------
    # Average
    # ------------------------------------------------------------

    for key in running:

        running[key] /= number_of_batches

    return running


# ================================================================
# VALIDATION
# ================================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running = {
        "total": 0.0,
        "occupancy": 0.0,
        "x": 0.0,
        "y": 0.0,
        "energy": 0.0,
    }

    number_of_batches = 0

    for batch in loader:

        momentum = batch[
            "momentum"
        ].to(device, non_blocking=True)

        refractive_index = batch[
            "refractive_index"
        ].to(device, non_blocking=True)

        batch["occupancy"] = batch[
            "occupancy"
        ].to(device, non_blocking=True)

        batch["x_tokens"] = batch[
            "x_tokens"
        ].to(device, non_blocking=True)

        batch["y_tokens"] = batch[
            "y_tokens"
        ].to(device, non_blocking=True)

        batch["energy_tokens"] = batch[
            "energy_tokens"
        ].to(device, non_blocking=True)

        outputs = model(
            momentum=momentum,
            refractive_index=refractive_index,
        )

        losses = criterion(
            outputs,
            batch
        )

        for key in running:

            running[key] += (
                losses[key]
                .item()
            )

        number_of_batches += 1

    for key in running:

        running[key] /= number_of_batches

    return running


# ================================================================
# CHECKPOINT
# ================================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    train_losses,
    val_losses,
):

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "train_losses":
            train_losses,

        "val_losses":
            val_losses,
    }

    torch.save(
        checkpoint,
        path
    )


# ================================================================
# MAIN
# ================================================================

def main():

    # ------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------

    set_seed(
        RANDOM_SEED
    )

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = get_device()

    # ------------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------------

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    print()
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    dataset = CherenkovDataset(
        H5_PATH
    )

    print(
        "Total events:",
        len(dataset)
    )

    # ------------------------------------------------------------
    # Train / validation split
    # ------------------------------------------------------------

    train_size = int(
        TRAIN_FRACTION * len(dataset)
    )

    val_size = (
        len(dataset)
        - train_size
    )

    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    print(
        "Training events:",
        len(train_dataset)
    )

    print(
        "Validation events:",
        len(val_dataset)
    )

    # ------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

        collate_fn=collate_cherenkov_generation
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

        collate_fn=collate_cherenkov_generation
    )

    print()
    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    config = GeneratorConfig()

    model = CherenkovGenerator(
        config
    ).to(device)

    print_parameter_count(
        model
    )

    print_expert_status(
        model
    )

    # ------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------

    criterion = CherenkovLoss(
        config
    ).to(device)

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )

    # ------------------------------------------------------------
    # Mixed precision scaler
    # ------------------------------------------------------------

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda")
    )

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    print()
    print("=" * 60)
    print("STARTING FOUNDATION TRAINING")
    print("=" * 60)

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print()
        print(
            "=" * 60
        )

        print(
            f"EPOCH {epoch}/{NUM_EPOCHS}"
        )

        print(
            "=" * 60
        )

        # --------------------------------------------------------
        # Train
        # --------------------------------------------------------

        train_losses = train_one_epoch(
            model=model,

            loader=train_loader,

            criterion=criterion,

            optimizer=optimizer,

            scaler=scaler,

            device=device,

            epoch=epoch
        )

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        val_losses = validate(
            model=model,

            loader=val_loader,

            criterion=criterion,

            device=device
        )

        # --------------------------------------------------------
        # Scheduler
        # --------------------------------------------------------

        scheduler.step()

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # --------------------------------------------------------
        # Print results
        # --------------------------------------------------------

        print()
        print(
            f"Epoch {epoch:03d} results:"
        )

        print(
            f"  Train total:     "
            f"{train_losses['total']:.6f}"
        )

        print(
            f"  Train occupancy: "
            f"{train_losses['occupancy']:.6f}"
        )

        print(
            f"  Train x:         "
            f"{train_losses['x']:.6f}"
        )

        print(
            f"  Train y:         "
            f"{train_losses['y']:.6f}"
        )

        print(
            f"  Train energy:    "
            f"{train_losses['energy']:.6f}"
        )

        print()

        print(
            f"  Val total:       "
            f"{val_losses['total']:.6f}"
        )

        print(
            f"  Val occupancy:   "
            f"{val_losses['occupancy']:.6f}"
        )

        print(
            f"  Val x:           "
            f"{val_losses['x']:.6f}"
        )

        print(
            f"  Val y:           "
            f"{val_losses['y']:.6f}"
        )

        print(
            f"  Val energy:      "
            f"{val_losses['energy']:.6f}"
        )

        print()

        print(
            f"  Learning rate:   "
            f"{current_lr:.8e}"
        )

        # --------------------------------------------------------
        # Save latest
        # --------------------------------------------------------

        save_checkpoint(
            LAST_CHECKPOINT,

            model,

            optimizer,

            scheduler,

            epoch,

            train_losses,

            val_losses
        )

        # --------------------------------------------------------
        # Save best
        # --------------------------------------------------------

        if (
            val_losses["total"]
            <
            best_validation_loss
        ):

            best_validation_loss = (
                val_losses["total"]
            )

            save_checkpoint(
                BEST_CHECKPOINT,

                model,

                optimizer,

                scheduler,

                epoch,

                train_losses,

                val_losses
            )

            print()
            print(
                "✓ New best model saved."
            )

            print(
                BEST_CHECKPOINT
            )

    # ------------------------------------------------------------
    # Finished
    # ------------------------------------------------------------

    print()
    print("=" * 60)
    print("FOUNDATION TRAINING COMPLETE")
    print("=" * 60)

    print()
    print(
        "Best validation loss:",
        best_validation_loss
    )

    print()
    print(
        "Best checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )

    print()
    print(
        "Last checkpoint:"
    )

    print(
        LAST_CHECKPOINT
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Experts 0 and 1 were trained."
    )

    print(
        "Experts 2 and 3 remained frozen."
    )


if __name__ == "__main__":

    main()