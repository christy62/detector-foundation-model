import h5py
import torch
from torch.utils.data import Dataset, DataLoader


N_MAX_HITS = 192


class CherenkovDataset(Dataset):
    """
    PyTorch Dataset for tokenized Geant4 Cherenkov events.

    Each event contains a variable number of photon hits.

    For generation training, the hits are sorted by descending
    photon energy and then placed into fixed slots.

    Returns:
        x_tokens
        y_tokens
        energy_tokens
        momentum
        refractive_index
    """

    def __init__(self, h5_path):

        self.h5_path = h5_path
        self.h5 = None

        # Read number of events without keeping the file open.
        with h5py.File(self.h5_path, "r") as f:
            self.n_events = len(f["momentum"])

    def _open_file(self):
        """
        Open HDF5 lazily.

        Important when DataLoader uses multiple workers.
        """

        if self.h5 is None:
            self.h5 = h5py.File(
                self.h5_path,
                "r"
            )

    def __len__(self):
        return self.n_events

    def __getitem__(self, index):

        self._open_file()

        start = self.h5["offsets"][index]
        end = self.h5["offsets"][index + 1]

        x = self.h5["x_tokens"][start:end]
        y = self.h5["y_tokens"][start:end]
        energy = self.h5["energy_tokens"][start:end]

        momentum = self.h5["momentum"][index]
        refractive_index = self.h5["refractive_index"][index]

        # Convert to tensors
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        energy = torch.tensor(energy, dtype=torch.long)

        # ---------------------------------------------------------
        # IMPORTANT:
        #
        # Sort photons by descending energy.
        #
        # This gives every photon a deterministic slot:
        #
        # slot 0 = highest-energy photon
        # slot 1 = second-highest-energy photon
        # ...
        # ---------------------------------------------------------

        if len(energy) > 0:

            order = torch.argsort(
                energy,
                descending=True
            )

            x = x[order]
            y = y[order]
            energy = energy[order]

        return {
            "x_tokens": x,
            "y_tokens": y,
            "energy_tokens": energy,

            "momentum": torch.tensor(
                momentum,
                dtype=torch.float32
            ),

            "refractive_index": torch.tensor(
                refractive_index,
                dtype=torch.float32
            )
        }


def collate_cherenkov_generation(batch):
    """
    Collate function for Cherenkov event generation.

    Converts variable-length events into fixed-size tensors
    with N_MAX_HITS = 192 slots.

    Token convention:

        0 = PAD / EMPTY

        x:
            1..128 = physical x bins 0..127

        y:
            1..128 = physical y bins 0..127

        energy:
            1..111 = physical energy bins 0..110

    Occupancy:

        0 = EMPTY
        1 = HIT

    Since the photons were sorted by descending energy in
    __getitem__, the first N slots correspond to real photons
    and the remaining slots are empty.
    """

    batch_size = len(batch)

    # ---------------------------------------------------------
    # Fixed-size tensors
    # ---------------------------------------------------------

    x_tokens = torch.zeros(
        (batch_size, N_MAX_HITS),
        dtype=torch.long
    )

    y_tokens = torch.zeros(
        (batch_size, N_MAX_HITS),
        dtype=torch.long
    )

    energy_tokens = torch.zeros(
        (batch_size, N_MAX_HITS),
        dtype=torch.long
    )

    occupancy = torch.zeros(
        (batch_size, N_MAX_HITS),
        dtype=torch.long
    )

    # Number of hits in every event
    lengths = torch.zeros(
        batch_size,
        dtype=torch.long
    )

    momenta = torch.empty(
        batch_size,
        dtype=torch.float32
    )

    refractive_indices = torch.empty(
        batch_size,
        dtype=torch.float32
    )

    # ---------------------------------------------------------
    # Fill batch
    # ---------------------------------------------------------

    for i, sample in enumerate(batch):

        length = len(sample["x_tokens"])

        # Safety check
        if length > N_MAX_HITS:
            raise ValueError(
                f"Event contains {length} hits, "
                f"but N_MAX_HITS={N_MAX_HITS}."
            )

        if length > 0:

            # Shift tokens by +1.
            #
            # Original:
            #   0..127 → x/y
            #   0..110 → energy
            #
            # New:
            #   1..128 → x/y
            #   1..111 → energy
            #
            # 0 remains reserved for EMPTY/PAD.

            x_tokens[i, :length] = (
                sample["x_tokens"] + 1
            )

            y_tokens[i, :length] = (
                sample["y_tokens"] + 1
            )

            energy_tokens[i, :length] = (
                sample["energy_tokens"] + 1
            )

            # These positions contain real photons
            occupancy[i, :length] = 1

        lengths[i] = length

        momenta[i] = sample["momentum"]

        refractive_indices[i] = (
            sample["refractive_index"]
        )

    return {
        "x_tokens": x_tokens,
        "y_tokens": y_tokens,
        "energy_tokens": energy_tokens,

        "occupancy": occupancy,

        "lengths": lengths,

        "momentum": momenta,
        "refractive_index": refractive_indices
    }


# ----------------------------------------------------------------
# Original collate function
# ----------------------------------------------------------------

def collate_cherenkov(batch):
    """
    Original variable-length collate function.

    Kept for inspection/debugging.

    For model training, use:

        collate_cherenkov_generation
    """

    batch_size = len(batch)

    lengths = torch.tensor(
        [
            len(sample["x_tokens"])
            for sample in batch
        ],
        dtype=torch.long
    )

    max_length = int(
        lengths.max().item()
    )

    if max_length == 0:
        max_length = 1

    x_padded = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long
    )

    y_padded = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long
    )

    energy_padded = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long
    )

    attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.bool
    )

    momenta = torch.empty(
        batch_size,
        dtype=torch.float32
    )

    refractive_indices = torch.empty(
        batch_size,
        dtype=torch.float32
    )

    for i, sample in enumerate(batch):

        length = len(sample["x_tokens"])

        if length > 0:

            x_padded[i, :length] = (
                sample["x_tokens"] + 1
            )

            y_padded[i, :length] = (
                sample["y_tokens"] + 1
            )

            energy_padded[i, :length] = (
                sample["energy_tokens"] + 1
            )

            attention_mask[i, :length] = True

        momenta[i] = sample["momentum"]

        refractive_indices[i] = (
            sample["refractive_index"]
        )

    return {
        "x_tokens": x_padded,
        "y_tokens": y_padded,
        "energy_tokens": energy_padded,

        "attention_mask": attention_mask,

        "lengths": lengths,

        "momentum": momenta,
        "refractive_index": refractive_indices
    }


# ----------------------------------------------------------------
# TEST
# ----------------------------------------------------------------

if __name__ == "__main__":

    dataset = CherenkovDataset(
        "/content/drive/MyDrive/tokenized_training.h5"
    )

    print("=" * 60)
    print("CHERENKOV GENERATION DATALOADER TEST")
    print("=" * 60)

    print(
        "Number of events:",
        len(dataset)
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_cherenkov_generation
    )

    batch = next(iter(loader))

    print()
    print("Batch shapes:")
    print(
        "x_tokens        :",
        batch["x_tokens"].shape
    )

    print(
        "y_tokens        :",
        batch["y_tokens"].shape
    )

    print(
        "energy_tokens   :",
        batch["energy_tokens"].shape
    )

    print(
        "occupancy       :",
        batch["occupancy"].shape
    )

    print(
        "lengths         :",
        batch["lengths"].shape
    )

    print(
        "momentum        :",
        batch["momentum"].shape
    )

    print(
        "refractive_index:",
        batch["refractive_index"].shape
    )

    print()
    print("Hit counts:")
    print(batch["lengths"])

    print()
    print("Momentum:")
    print(batch["momentum"])

    print()
    print("Refractive index:")
    print(batch["refractive_index"])

    # ---------------------------------------------------------
    # Verify fixed size
    # ---------------------------------------------------------

    assert batch["x_tokens"].shape == (
        32,
        N_MAX_HITS
    )

    assert batch["y_tokens"].shape == (
        32,
        N_MAX_HITS
    )

    assert batch["energy_tokens"].shape == (
        32,
        N_MAX_HITS
    )

    assert batch["occupancy"].shape == (
        32,
        N_MAX_HITS
    )

    # ---------------------------------------------------------
    # Verify occupancy
    # ---------------------------------------------------------

    assert torch.all(
        batch["occupancy"].sum(dim=1)
        == batch["lengths"]
    )

    # ---------------------------------------------------------
    # Verify padding
    # ---------------------------------------------------------

    for i in range(32):

        length = batch["lengths"][i].item()

        if length < N_MAX_HITS:

            assert torch.all(
                batch["occupancy"][
                    i,
                    length:
                ] == 0
            )

            assert torch.all(
                batch["x_tokens"][
                    i,
                    length:
                ] == 0
            )

            assert torch.all(
                batch["y_tokens"][
                    i,
                    length:
                ] == 0
            )

            assert torch.all(
                batch["energy_tokens"][
                    i,
                    length:
                ] == 0
            )

    # ---------------------------------------------------------
    # Verify token ranges
    # ---------------------------------------------------------

    assert (
        batch["x_tokens"].min().item() >= 0
    )

    assert (
        batch["x_tokens"].max().item() <= 128
    )

    assert (
        batch["y_tokens"].min().item() >= 0
    )

    assert (
        batch["y_tokens"].max().item() <= 128
    )

    assert (
        batch["energy_tokens"].min().item() >= 0
    )

    assert (
        batch["energy_tokens"].max().item() <= 111
    )

    print()
    print("Token ranges:")
    print(
        "x:",
        batch["x_tokens"].min().item(),
        "to",
        batch["x_tokens"].max().item()
    )

    print(
        "y:",
        batch["y_tokens"].min().item(),
        "to",
        batch["y_tokens"].max().item()
    )

    print(
        "energy:",
        batch["energy_tokens"].min().item(),
        "to",
        batch["energy_tokens"].max().item()
    )

    print()
    print("Maximum hits in batch:")
    print(
        batch["lengths"].max().item()
    )

    print()
    print("DATALOADER TEST PASSED")