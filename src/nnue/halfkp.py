import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
import torch.optim as optim
import chess
import argparse
from tqdm import tqdm
import numpy as np
import os
import pickle
import gc
import bisect

# print(f'GPU available: {torch.cuda.is_available()}')
# print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')

# --- Helper: Forced Mate Evaluation Parser ---

def parse_evaluation(eval_str, mate_base=3000):
    """
    Parse evaluation string with capped mate scores.
    mate_base=3000 ensures mate scores don't dominate training.
    """
    s = str(eval_str).lstrip('\ufeff').strip()
    if s.startswith("#"):
        mate_part = s[1:].replace("+", "").strip()
        try:
            moves_to_mate = int(mate_part)
        except ValueError:
            moves_to_mate = 0
        val = mate_base - abs(moves_to_mate)
        if mate_part.startswith("-"):
            val = -val
        return float(val)
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Invalid evaluation string: '{eval_str}'")


def is_valid_eval(eval_str):
    try:
        parse_evaluation(eval_str)
        return True
    except:
        return False

# --- Dataset with Precomputed HalfKP Indices ---
# HalfKP encodes (king_square, piece_square, piece_type/color).
# 10 non-king piece types (5 types x 2 colors), 64 piece squares, 64 king squares.
# Index = king_sq * (64 * 10) + piece_sq * 10 + piece_type_offset
NUM_NONKING = 10
PIECES_PER_KING = 64 * NUM_NONKING   # 640
TABLE_SIZE = 64 * PIECES_PER_KING     # 40960

piece_to_idx = {
    (True,  chess.PAWN):   0,
    (True,  chess.KNIGHT): 1,
    (True,  chess.BISHOP): 2,
    (True,  chess.ROOK):   3,
    (True,  chess.QUEEN):  4,
    (False, chess.PAWN):   5,
    (False, chess.KNIGHT): 6,
    (False, chess.BISHOP): 7,
    (False, chess.ROOK):   8,
    (False, chess.QUEEN):  9,
}


def halfkp_indices_for_fen(fen):
    board = chess.Board(fen)
    idx0, idx1 = [], []
    for view in (0, 1):
        king_sq = board.king(bool(view))
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.piece_type != chess.KING:
                offset = piece_to_idx[(piece.color, piece.piece_type)]
                idx = king_sq * PIECES_PER_KING + sq * NUM_NONKING + offset
                (idx0 if view == 0 else idx1).append(idx)
    return idx0, idx1

class ChessDatasetHalfKP(Dataset):
    """
    Stores HalfKP indices as flat numpy arrays with offset tables per chunk.
    Each chunk file contains: flat0, off0, flat1, off1 (all pure int32/int64 numpy).
    Targets are stored as a separate .npy file and memory-mapped.

    RAM during processing: <4 GB (targets written directly to disk via memmap).
    RAM during training: ~1.2 GB (mmap'd targets) + ~150 MB per cached chunk.
    """
    def __init__(self, csv_file, scale=100.0, use_cache=True):
        # Persistent chunk directory lives next to the CSV
        chunk_dir = csv_file.rsplit('.', 1)[0] + '_halfkp_v6'
        meta_file = os.path.join(chunk_dir, 'meta.pkl')
        targets_file = os.path.join(chunk_dir, 'targets.npy')

        # ── Try loading from cached chunks ──
        if use_cache and os.path.exists(meta_file):
            print(f"Loading cached dataset from {chunk_dir}...")
            with open(meta_file, 'rb') as f:
                meta = pickle.load(f)
            self.targets = np.load(targets_file, mmap_mode='r')
            self.chunk_offsets = meta['chunk_offsets']
            self.max0 = meta['max0']
            self.max1 = meta['max1']
            self.num_chunks = meta['num_chunks']
            self.chunk_dir = chunk_dir
            self._chunk_cache = {}
            self._cache_limit = 4
            print(f"[OK] Loaded {len(self.targets)} samples from cache ({self.num_chunks} chunks)")
            return

        # ── Process CSV in chunks ──
        from multiprocessing import Pool, cpu_count

        os.makedirs(chunk_dir, exist_ok=True)
        self.max0 = 0
        self.max1 = 0
        chunk_num = 0
        row_count = 0
        chunk_sizes = []

        # Pre-allocate targets as a memory-mapped file (avoids 11 GB Python list)
        total_est = 316072343
        targets_mmap = np.lib.format.open_memmap(
            targets_file, mode='w+', dtype=np.float32, shape=(total_est,))
        target_pos = 0

        print(f"Reading CSV from {csv_file} and computing HalfKP indices (streaming to disk)...")

        csv_reader = pd.read_csv(csv_file, chunksize=10000000)
        total_chunks = (total_est // 10000000) + 1
        for df_chunk in tqdm(csv_reader, desc="Reading CSV chunks", unit="chunk", total=total_chunks):
            tqdm.write(f"Processing chunk {chunk_num + 1}...")

            df_chunk.columns = df_chunk.columns.str.strip()

            chunk_fens = df_chunk['FEN'].tolist()

            # Write targets directly to memmap (no Python list accumulation)
            chunk_targets = np.array(
                [parse_evaluation(v)/scale for v in df_chunk['Evaluation']],
                dtype=np.float32)
            targets_mmap[target_pos:target_pos + len(chunk_targets)] = chunk_targets
            target_pos += len(chunk_targets)
            del chunk_targets

            with Pool(processes=cpu_count()) as pool:
                results = list(tqdm(pool.imap(halfkp_indices_for_fen, chunk_fens, chunksize=10000),
                                   total=len(chunk_fens),
                                   desc=f"Computing indices for chunk {chunk_num + 1}",
                                   unit="fen",
                                   leave=False))

            chunk_idx0, chunk_idx1 = zip(*results)
            chunk_idx0 = list(chunk_idx0)
            chunk_idx1 = list(chunk_idx1)

            for i0, i1 in zip(chunk_idx0, chunk_idx1):
                self.max0 = max(self.max0, len(i0))
                self.max1 = max(self.max1, len(i1))

            # Convert jagged lists to flat arrays + offset tables
            # e.g. [[1,2,3],[4,5],[6]] -> flat=[1,2,3,4,5,6], off=[0,3,5,6]
            lengths0 = np.array([len(r) for r in chunk_idx0], dtype=np.int64)
            off0 = np.zeros(len(chunk_idx0) + 1, dtype=np.int64)
            np.cumsum(lengths0, out=off0[1:])
            flat0 = np.concatenate([np.array(r, dtype=np.int32) for r in chunk_idx0])

            lengths1 = np.array([len(r) for r in chunk_idx1], dtype=np.int64)
            off1 = np.zeros(len(chunk_idx1) + 1, dtype=np.int64)
            np.cumsum(lengths1, out=off1[1:])
            flat1 = np.concatenate([np.array(r, dtype=np.int32) for r in chunk_idx1])

            # Save as individual .npy files so they can be memory-mapped
            base = os.path.join(chunk_dir, f'chunk_{chunk_num}')
            np.save(f'{base}_flat0.npy', flat0)
            np.save(f'{base}_off0.npy', off0)
            np.save(f'{base}_flat1.npy', flat1)
            np.save(f'{base}_off1.npy', off1)

            chunk_sizes.append(len(chunk_idx0))
            chunk_num += 1
            row_count += len(chunk_fens)

            del df_chunk, chunk_fens, results, chunk_idx0, chunk_idx1
            del flat0, off0, flat1, off1, lengths0, lengths1
            gc.collect()

        # Truncate memmap to actual size (in case estimate was off)
        targets_mmap.flush()
        del targets_mmap
        if row_count != total_est:
            tmp = np.load(targets_file, mmap_mode='r')[:row_count].copy()
            np.save(targets_file, tmp)
            del tmp

        self.targets = np.load(targets_file, mmap_mode='r')

        # Build cumulative offsets: [0, size0, size0+size1, ...]
        self.chunk_offsets = [0]
        for s in chunk_sizes:
            self.chunk_offsets.append(self.chunk_offsets[-1] + s)
        self.num_chunks = chunk_num
        self.chunk_dir = chunk_dir

        print(f"[OK] Extracted {row_count} FENs in {chunk_num} chunks")
        print(f"  Max view0 features: {self.max0}, Max view1 features: {self.max1}")

        # Save metadata (tiny — just offsets + a few ints, no targets)
        if use_cache:
            print(f"Saving metadata to {meta_file}...")
            with open(meta_file, 'wb') as f:
                pickle.dump({
                    'chunk_offsets': self.chunk_offsets,
                    'max0': self.max0,
                    'max1': self.max1,
                    'num_chunks': self.num_chunks,
                }, f)
            print(f"[OK] Cache saved ({len(self.targets)} samples)")

        self._chunk_cache = {}
        self._cache_limit = 4

    def _load_chunk(self, chunk_id):
        """Load a chunk via memory-mapping (uses virtually no heap RAM)."""
        if chunk_id not in self._chunk_cache:
            base = os.path.join(self.chunk_dir, f'chunk_{chunk_id}')
            loaded = (
                np.load(f'{base}_flat0.npy', mmap_mode='r'),
                np.load(f'{base}_off0.npy',  mmap_mode='r'),
                np.load(f'{base}_flat1.npy', mmap_mode='r'),
                np.load(f'{base}_off1.npy',  mmap_mode='r'),
            )
            if len(self._chunk_cache) >= self._cache_limit:
                oldest = next(iter(self._chunk_cache))
                del self._chunk_cache[oldest]
            self._chunk_cache[chunk_id] = loaded
        return self._chunk_cache[chunk_id]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # O(log n) lookup: which chunk contains this sample?
        chunk_id = bisect.bisect_right(self.chunk_offsets, idx) - 1
        local_idx = idx - self.chunk_offsets[chunk_id]

        flat0, off0, flat1, off1 = self._load_chunk(chunk_id)

        # numpy slice into cached arrays
        row0 = flat0[off0[local_idx]:off0[local_idx + 1]]
        row1 = flat1[off1[local_idx]:off1[local_idx + 1]]

        return (row0, row1, self.targets[idx])

# --- Accumulator and Sparse HalfKP NNUE Model ---
# With TABLE_SIZE=40960, embedding tables are much larger, so reduce hidden size
# to keep parameter count manageable while maintaining expressiveness.
HIDDEN_SIZE = 1024
MLP_HIDDEN = 64

# Parameter count:
#   Embeddings: 2 * 40960 * 256 = ~21M params
#   MLP: 512->32->32->1 = ~17K params
#   Total: ~21M (vs ~168M with HIDDEN_SIZE=2048)

class SparseAccumulator:
    """
    Maintains a sparse accumulator state for HalfKP features.
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.acc0 = None
        self.acc1 = None

    def reset(self, idx0, idx1, emb0, emb1):
        # idx0, idx1: LongTensors of shape (batch, L0) and (batch, L1)
        sum0 = emb0(idx0).sum(dim=1)      # (batch, H)
        sum1 = emb1(idx1).sum(dim=1)
        self.acc0 = torch.clamp(sum0, min=0)
        self.acc1 = torch.clamp(sum1, min=0)
        return torch.cat([self.acc0, self.acc1], dim=1)

    def update(self, removed0, added0, removed1, added1, emb0, emb1):
        # removed*/added*: indices removed/added since last state
        delta0 = emb0(added0).sum(dim=1) - emb0(removed0).sum(dim=1)
        delta1 = emb1(added1).sum(dim=1) - emb1(removed1).sum(dim=1)
        self.acc0 = torch.clamp(self.acc0 + delta0, min=0)
        self.acc1 = torch.clamp(self.acc1 + delta1, min=0)
        return torch.cat([self.acc0, self.acc1], dim=1)

class HalfKP_NNUE(nn.Module):
    def __init__(self, max0=0, max1=0):
        super().__init__()
        # Embedding tables — TABLE_SIZE=40960 entries, one per (king_sq, piece_sq, piece_type)
        self.emb0 = nn.Embedding(TABLE_SIZE, HIDDEN_SIZE)
        self.emb1 = nn.Embedding(TABLE_SIZE, HIDDEN_SIZE)
        # Initialize embeddings small
        nn.init.normal_(self.emb0.weight, std=0.01)
        nn.init.normal_(self.emb1.weight, std=0.01)
        # MLP: concat both views → hidden layers → scalar output
        input_size = 2 * HIDDEN_SIZE  # 512
        self.fc2 = nn.Linear(input_size, MLP_HIDDEN)
        self.fc3 = nn.Linear(MLP_HIDDEN, MLP_HIDDEN)
        self.fc_out = nn.Linear(MLP_HIDDEN, 1)
        # Sparse accumulator instance (not a parameter)
        self.accumulator = SparseAccumulator(HIDDEN_SIZE)

    def forward(self, idx0_batch, idx1_batch):
        """Traceable forward pass for TorchScript export (no mutable accumulator)."""
        sum0 = self.emb0(idx0_batch).sum(dim=1)   # (batch, H)
        sum1 = self.emb1(idx1_batch).sum(dim=1)
        h = torch.cat([torch.clamp(sum0, min=0), torch.clamp(sum1, min=0)], dim=1)
        return self._mlp(h)

    def forward_reset(self, idx0_batch, idx1_batch):
        # initial full reset
        h = self.accumulator.reset(idx0_batch, idx1_batch, self.emb0, self.emb1)
        return self._mlp(h)

    def forward_update(self, rem0, add0, rem1, add1):
        # incremental update: indices of removed/added per view
        h = self.accumulator.update(rem0, add0, rem1, add1, self.emb0, self.emb1)
        return self._mlp(h)

    def _mlp(self, h):
        x = F.relu(self.fc2(h))
        x = F.relu(self.fc3(x))
        return self.fc_out(x).squeeze(-1)

# Training & Export Utilities (unchanged, except adapt to new forward_reset)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_halfkp(batch):
    """Efficient batching: pad jagged indices once per batch."""
    i0_list, i1_list, targets = zip(*batch)
    max0 = max(len(x) for x in i0_list)
    max1 = max(len(x) for x in i1_list)

    i0_batch = np.zeros((len(batch), max0), dtype=np.int64)
    i1_batch = np.zeros((len(batch), max1), dtype=np.int64)

    for j, (i0, i1) in enumerate(zip(i0_list, i1_list)):
        i0_batch[j, :len(i0)] = i0
        i1_batch[j, :len(i1)] = i1

    return (
        torch.from_numpy(i0_batch),
        torch.from_numpy(i1_batch),
        torch.tensor(np.array(targets), dtype=torch.float32)
    )

def chunk_batch_generator(ds, batch_size, subset_indices=None):
    """
    Vectorized chunk-streaming batch generator with background prefetching.
    Loads/pads the next chunk in a background thread while GPU trains on the current one.
    Returns (generator_factory, num_batches).
    """
    from threading import Thread
    from queue import Queue

    if subset_indices is not None:
        indices = np.sort(np.array(subset_indices))
    else:
        indices = np.arange(len(ds))

    # Group indices by chunk for sequential disk access
    chunk_groups = []
    for chunk_id in range(ds.num_chunks):
        lo = ds.chunk_offsets[chunk_id]
        hi = ds.chunk_offsets[chunk_id + 1]
        mask = (indices >= lo) & (indices < hi)
        chunk_indices = indices[mask]
        if len(chunk_indices) > 0:
            chunk_groups.append((chunk_id, chunk_indices))

    total_samples = len(indices)
    num_batches = (total_samples + batch_size - 1) // batch_size

    def _pad_chunk_vectorized(flat, off, local_indices, max_feat):
        """
        Fully vectorized padding — no Python loops over samples.
        """
        n = len(local_indices)
        starts = np.array(off[local_indices])
        ends = np.array(off[local_indices + 1])
        lengths = ends - starts

        actual_max = int(lengths.max()) if n > 0 else 0
        pad_width = min(max_feat, actual_max)

        result = np.zeros((n, pad_width), dtype=np.int64)

        total_elems = int(lengths.sum())
        if total_elems == 0:
            return result

        # Row indices: [0,0,0, 1,1, 2,2,2,2, ...]
        row_idx = np.repeat(np.arange(n, dtype=np.int64), lengths)

        # Column indices: [0,1,2, 0,1, 0,1,2,3, ...] — fully vectorized
        # Build a cumulative counter that resets at each sample boundary
        col_idx = np.arange(total_elems, dtype=np.int64)
        # Subtract the cumulative start of each sample's run
        offsets_per_elem = np.repeat(np.concatenate([[0], np.cumsum(lengths[:-1])]), lengths)
        col_idx -= offsets_per_elem

        # Source indices into flat array — fully vectorized
        src_idx = np.arange(total_elems, dtype=np.int64)
        sample_flat_starts = np.repeat(starts, lengths)
        sample_run_starts = np.repeat(np.concatenate([[0], np.cumsum(lengths[:-1])]), lengths)
        src_idx = sample_flat_starts + (src_idx - sample_run_starts)

        result[row_idx, col_idx] = flat[src_idx]
        return result

    def _load_and_pad_chunk(chunk_id, chunk_idx):
        """Load one chunk from disk and pad it. Runs in background thread."""
        base = os.path.join(ds.chunk_dir, f'chunk_{chunk_id}')
        flat0 = np.load(f'{base}_flat0.npy', mmap_mode='r')
        off0  = np.load(f'{base}_off0.npy',  mmap_mode='r')
        flat1 = np.load(f'{base}_flat1.npy', mmap_mode='r')
        off1  = np.load(f'{base}_off1.npy',  mmap_mode='r')
        lo = ds.chunk_offsets[chunk_id]

        local = chunk_idx - lo
        padded0 = _pad_chunk_vectorized(flat0, off0, local, ds.max0)
        padded1 = _pad_chunk_vectorized(flat1, off1, local, ds.max1)
        targets = np.array(ds.targets[chunk_idx], dtype=np.float32)
        return padded0, padded1, targets

    def _gen():
        carry_i0 = carry_i1 = carry_y = None

        # Prefetch queue: background thread loads next chunk while we yield batches
        prefetch_q = Queue(maxsize=1)

        def _prefetch_worker():
            for chunk_id, chunk_idx in chunk_groups:
                result = _load_and_pad_chunk(chunk_id, chunk_idx)
                prefetch_q.put(result)
            prefetch_q.put(None)  # sentinel

        worker = Thread(target=_prefetch_worker, daemon=True)
        worker.start()

        while True:
            chunk_data = prefetch_q.get()
            if chunk_data is None:
                break
            padded0, padded1, targets = chunk_data

            # Prepend carry-over from previous chunk
            if carry_i0 is not None:
                padded0 = _vstack_pad(carry_i0, padded0)
                padded1 = _vstack_pad(carry_i1, padded1)
                targets = np.concatenate([carry_y, targets])
                carry_i0 = carry_i1 = carry_y = None

            # Yield full batches
            n = len(targets)
            pos = 0
            while pos + batch_size <= n:
                yield (
                    torch.from_numpy(padded0[pos:pos+batch_size].copy()),
                    torch.from_numpy(padded1[pos:pos+batch_size].copy()),
                    torch.from_numpy(targets[pos:pos+batch_size].copy()),
                )
                pos += batch_size

            # Save remainder as carry-over
            if pos < n:
                carry_i0 = padded0[pos:].copy()
                carry_i1 = padded1[pos:].copy()
                carry_y  = targets[pos:].copy()

        worker.join()

        # Final partial batch
        if carry_i0 is not None and len(carry_i0) > 0:
            yield (
                torch.from_numpy(carry_i0),
                torch.from_numpy(carry_i1),
                torch.from_numpy(carry_y),
            )

    return _gen, num_batches

def _vstack_pad(a, b):
    """Vertically stack two 2D arrays, zero-padding the narrower one."""
    if a.shape[1] == b.shape[1]:
        return np.vstack([a, b])
    max_cols = max(a.shape[1], b.shape[1])
    if a.shape[1] < max_cols:
        a = np.pad(a, ((0, 0), (0, max_cols - a.shape[1])))
    elif b.shape[1] < max_cols:
        b = np.pad(b, ((0, 0), (0, max_cols - b.shape[1])))
    return np.vstack([a, b])

def train_model(csv_file,
                epochs: int = 100,
                batch_size: int = 8192,
                lr: float = 2e-4,
                l2: float = 1e-7,
                sample_frac: float = 1.0):
    """
    Train HalfKP NNUE model.
    :param sample_frac: Fraction of dataset to use (0.0-1.0). Use <1.0 for faster iteration.
    """
    from torch.amp import autocast, GradScaler

    print("=" * 70)
    print("TRAINING HALFKP NNUE MODEL")
    print("=" * 70)

    print("\n[1/6] Loading dataset...")
    ds = ChessDatasetHalfKP(csv_file)

    # Subsample if requested: pick random indices but keep the raw dataset
    n_full = len(ds)
    if sample_frac < 1.0:
        import random
        n_sample = max(1, int(n_full * sample_frac))
        all_indices = np.array(random.sample(range(n_full), n_sample))
        print(f"  Subsampled to {n_sample}/{n_full} samples ({sample_frac*100:.0f}%)")
    else:
        n_sample = n_full
        all_indices = np.arange(n_full)

    n = n_sample
    n_train = int(0.50 * n)
    n_val   = int(0.25 * n)
    n_test  = n - n_train - n_val
    print(f"[OK] Dataset loaded: {n} total samples")

    print(f"\n[2/6] Splitting data...")
    # Shuffle then split indices (equivalent to random_split)
    np.random.shuffle(all_indices)
    train_idx = all_indices[:n_train]
    val_idx   = all_indices[n_train:n_train + n_val]
    test_idx  = all_indices[n_train + n_val:]
    print(f"  • Training:   {n_train} samples ({100*n_train/n:.1f}%)")
    print(f"  • Validation: {n_val} samples ({100*n_val/n:.1f}%)")
    print(f"  • Test:       {n_test} samples ({100*n_test/n:.1f}%)")

    print(f"\n[3/6] Creating chunk-streaming data loaders (batch_size={batch_size})...")
    train_gen, train_batches = chunk_batch_generator(ds, batch_size, train_idx)
    val_gen,   val_batches   = chunk_batch_generator(ds, batch_size, val_idx)
    test_gen,  test_batches  = chunk_batch_generator(ds, batch_size, test_idx)
    print(f"[OK] Data loaders created")
    print(f"  • Training batches:   {train_batches}")
    print(f"  • Validation batches: {val_batches}")
    print(f"  • Test batches:       {test_batches}")

    print(f"\n[4/6] Initializing model on {DEVICE}...")
    model   = HalfKP_NNUE().to(DEVICE)
    opt     = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.SmoothL1Loss()
    use_amp = DEVICE.type == 'cuda'
    scaler  = GradScaler(device=DEVICE if use_amp else 'cpu', enabled=use_amp)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
    print(f"[OK] Model initialized")
    print(f"  • Device: {DEVICE}")
    print(f"  • Mixed precision: {use_amp}")
    print(f"  • Learning rate: {lr}")
    print(f"  • L2 regularization: {l2}")
    print(f"  • LR scheduler: StepLR (decay 0.5x every 3 epochs)")

    best_val_loss = float('inf')
    best_model_path = 'halfkp_best.pth'

    print(f"\n[5/6] Starting training for {epochs} epochs...")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        # ——— training ———
        model.train()
        train_bar = tqdm(train_gen(), total=train_batches,
                         desc=f"Epoch {epoch}/{epochs} [TRAIN]",
                         unit="batch",
                         ncols=100)
        train_loss_total = 0.0
        for x0, x1, y in train_bar:
            x0, x1, y = x0.to(DEVICE, non_blocking=True), x1.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            with autocast(device_type=DEVICE.type, enabled=use_amp):
                pred = model.forward_reset(x0, x1)
                loss = loss_fn(pred, y)

            train_loss_total += loss.item() * y.size(0)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss_avg = train_loss_total / n_train
        print(f"  [OK] Training Loss (avg): {train_loss_avg:.4f}")

        # ——— validation ———
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x0, x1, y in tqdm(val_gen(), total=val_batches, desc=f"Epoch {epoch}/{epochs} [VALID]", unit="batch", ncols=100, leave=False):
                x0, x1, y = x0.to(DEVICE, non_blocking=True), x1.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                with autocast(device_type=DEVICE.type, enabled=use_amp):
                    pred = model.forward_reset(x0, x1)
                    batch_loss = loss_fn(pred, y).item()
                val_loss += batch_loss * y.size(0)
        val_loss /= n_val
        print(f"  [OK] Validation Loss: {val_loss:.4f}")

        # <-- save best model if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  *** BEST MODEL SAVED to {best_model_path} (Val Loss: {val_loss:.4f}) ***")
        else:
            print(f"  (best so far: {best_val_loss:.4f})")
        scheduler.step()
        print()

    # ——— final testing ———
    print("=" * 70)
    print(f"[6/6] Final evaluation on test set...")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x0, x1, y in tqdm(test_gen(), total=test_batches, desc="[TEST]", unit="batch", ncols=100):
            x0, x1, y = x0.to(DEVICE, non_blocking=True), x1.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type, enabled=use_amp):
                pred = model.forward_reset(x0, x1)
                batch_loss = loss_fn(pred, y).item()
            test_loss += batch_loss * y.size(0)
    test_loss /= n_test
    print(f"[OK] Test Loss: {test_loss:.4f}")
    print("=" * 70)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train'], default='train')
    parser.add_argument('--data', default='./data/combined_chessData.csv')
    parser.add_argument('--sample', type=float, default=1.0,
                        help='Fraction of dataset to use (0.0-1.0). Use <1.0 for faster iteration.')
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    if args.mode == 'train':
        train_model(args.data, epochs=args.epochs, batch_size=args.batch_size,
                    sample_frac=args.sample)
