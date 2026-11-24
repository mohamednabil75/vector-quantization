import math
import numpy as np
from PIL import Image


# 7. Read the codebook file
def load_codebook(codebook_file):
    with open(codebook_file, "r") as f:
        # Read metadata
        bh, bw, K, orig_h, orig_w = map(int, f.readline().split())

        # Read codevectors
        codebook = []
        for _ in range(K):
            vector = list(map(int, f.readline().split()))
            codebook.append(vector)

    return np.array(codebook, dtype=np.uint8), bh, bw, K, orig_h, orig_w


# 8. Read the compressed binary stream and decode labels
def load_labels(stream_file, K, n_blocks):
    # Number of bits needed to represent one codevector index
    n = max(1, math.ceil(math.log2(K)))

    with open(stream_file, "rb") as f:
        data = f.read()

    # Convert bytes into a binary string
    bits = "".join(format(byte, "08b") for byte in data)

    # Read only the bits belonging to actual labels
    bits = bits[: n_blocks * n]

    # Convert every n bits into an integer index
    labels = [int(bits[i : i + n], 2) for i in range(0, len(bits), n)]

    return np.array(labels, dtype=int)


# 9. Reconstruct the image from the decoded labels
def decompress(codebook_file, stream_file, output_image):
    # Read codebook and original image information
    codebook, bh, bw, K, orig_h, orig_w = load_codebook(codebook_file)

    # Calculate padded image dimensions
    padded_h = ((orig_h + bh - 1) // bh) * bh
    padded_w = ((orig_w + bw - 1) // bw) * bw

    # Calculate number of blocks
    n_blocks = (padded_h // bh) * (padded_w // bw)

    # Decode the binary stream
    labels = load_labels(stream_file, K, n_blocks)

    # Create an empty padded image
    reconstructed = np.zeros((padded_h, padded_w), dtype=np.uint8)

    # Place each decoded codevector in its correct position
    block_index = 0
    for r in range(0, padded_h, bh):
        for c in range(0, padded_w, bw):
            vector = codebook[labels[block_index]]
            block = vector.reshape(bh, bw)
            reconstructed[r : r + bh, c : c + bw] = block
            block_index += 1

    # Remove padding to restore original image dimensions
    reconstructed = np.clip(reconstructed[:orig_h, :orig_w], 0, 255).astype(np.uint8)

    # Save reconstructed image
    Image.fromarray(reconstructed).save(output_image)

    print("Decompression completed successfully!")
    print("Reconstructed image saved as:", output_image)

    return reconstructed