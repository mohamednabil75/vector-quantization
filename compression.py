import numpy as np
from PIL import Image

print(-4%5)
# 1. Add padding so the image size is a multiple of the block size
def add_padding(img, bh, bw):
    pad_h = (bh - img.shape[0] % bh) % bh      
    pad_w = (bw - img.shape[1] % bw) % bw
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")


# 2. Divide the image into blocks (each block becomes one vector)
def split_into_blocks(img, bh, bw):
    blocks = []
    for r in range(0, img.shape[0], bh):
        for c in range(0, img.shape[1], bw):
            blocks.append(img[r:r + bh, c:c + bw].flatten())
    return np.array(blocks, dtype=float)


# 3. For every block find the index of the nearest codevector (distance = sum |x - y|)
def find_nearest(blocks, codebook):
    labels = []
    for b in blocks:
        distances = [np.sum(np.abs(b - c)) for c in codebook]
        labels.append(int(np.argmin(distances)))
    return np.array(labels)


# 4. New centroid of each cluster = average of the blocks inside it
def update_centroids(blocks, labels, codebook):
    new = codebook.copy()
    for k in range(len(codebook)):
        members = blocks[labels == k]
        if len(members) > 0:              # an empty cluster keeps its old centroid
            new[k] = members.mean(axis=0)
    return new


# 5. LBG algorithm with splitting
def lbg(blocks, K):
    codebook = blocks.mean(axis=0, keepdims=True)      # start: average of all blocks
    labels = np.zeros(len(blocks), dtype=int)

    while len(codebook) < K:
        # split every centroid X into (X-1) and (X+1)
        # (if K is not a power of 2, only the biggest clusters are split in the last round)
        counts = np.bincount(labels, minlength=len(codebook))
        how_many = min(len(codebook), K - len(codebook))
        to_split = list(np.argsort(-counts)[:how_many])
        new_codebook = []
        for i, c in enumerate(codebook):
            if i in to_split:
                new_codebook.append(np.ceil(c - 1))
                new_codebook.append(np.floor(c + 1))
            else:
                new_codebook.append(c)
        codebook = np.array(new_codebook)

        labels = find_nearest(blocks, codebook)        # nearest centroid for each block
        codebook = update_centroids(blocks, labels, codebook)

    # repeat until no block changes its cluster
    while True:
        new_labels = find_nearest(blocks, codebook)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        codebook = update_centroids(blocks, labels, codebook)

    # pixels are 8-bit integers -> round the codebook and find the final labels
    codebook = np.rint(codebook).astype(int)
    labels = find_nearest(blocks, codebook)
    return codebook, labels


# 6. Save the codebook file and the compressed file (binary stream)
def save_files(codebook, labels, bh, bw, orig_h, orig_w, codebook_file, stream_file):
  
    with open(codebook_file, "w") as f:
        f.write(f"{bh} {bw} {len(codebook)} {orig_h} {orig_w}\n")
        for c in codebook:
            f.write(" ".join(str(v) for v in c) + "\n")

    # compressed file: every label written with n bits, all bits packed into bytes
    n = max(1, int(np.ceil(np.log2(len(codebook)))))
    bits = "".join(format(int(l), f"0{n}b") for l in labels)
    bits += "0" * ((-len(bits)) % 8)                   # fill the last byte
    data = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
    with open(stream_file, "wb") as f:
        f.write(data)
    return n


def compression_ratio(orig_h, orig_w, bh, bw, K, n_blocks, n):
    original = orig_h * orig_w * 8
    codebook_bits = K * bh * bw * 8
    label_bits = n_blocks * n
    return original, codebook_bits, label_bits, original / (codebook_bits + label_bits)


# ----------------------------- main -----------------------------
if __name__ == "__main__":
    # path = input("Image path: ")
    # bh = int(input("Block height: "))
    # bw = int(input("Block width: "))
    # K = int(input("Number of vectors in codebook: "))
    path="peppers.png"
    bh=2
    bw=2
    K=4    

    img = np.array(Image.open(path).convert("L"))      
    orig_h, orig_w = img.shape

    padded = add_padding(img, bh, bw)
    blocks = split_into_blocks(padded, bh, bw)
    codebook, labels = lbg(blocks, K)
    n = save_files(codebook, labels, bh, bw, orig_h, orig_w, "codebook.txt", "compressed.bin")

    original, cb_bits, lab_bits, ratio = compression_ratio(orig_h, orig_w, bh, bw, K, len(blocks), n)
    print("Original size :", original, "bits")
    print("Codebook size :", cb_bits, "bits")
    print("Labels size   :", lab_bits, "bits")