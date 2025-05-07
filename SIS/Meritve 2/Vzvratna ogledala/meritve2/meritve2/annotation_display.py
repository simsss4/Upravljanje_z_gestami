import numpy as np
import os
import glob

DATA_DIR = 'sequences_data'

def load_sequences(data_dir):
    sequences = []
    labels = []
    
    for label_dir in glob.glob(os.path.join(data_dir, '*')):
        if os.path.isdir(label_dir):
            label = os.path.basename(label_dir)
            for npy_file in glob.glob(os.path.join(label_dir, '*.npy')):
                sequence = np.load(npy_file)
                sequences.append(sequence)
                labels.append(label)
                print(f"Naložena sekvenca: {npy_file}, Oblika: {sequence.shape}, Oznaka: {label}")
    
    return sequences, labels

def print_landmarks(sequence, sequence_idx):
    print(f"\nPrikaz koordinat za sekvenco {sequence_idx}:")
    for frame_idx, landmarks in enumerate(sequence):
        print(f"\nFrame {frame_idx}:")
        for i in range(0, len(landmarks), 3):
            x, y, z = landmarks[i], landmarks[i+1], landmarks[i+2]
            print(f"  Točka {i//3}: x={x:.4f}, y={y:.4f}, z={z:.4f}")

def main():
    sequences, labels = load_sequences(DATA_DIR)
    
    if not sequences:
        print("Ni najdenih .npy datotek v mapi:", DATA_DIR)
        return
    
    print(f"\nSkupno število naloženih sekvenc: {len(sequences)}")
    
    for idx, (sequence, label) in enumerate(zip(sequences, labels)):
        print(f"\nSekvenca {idx}, Oznaka: {label}, Število okvirjev: {sequence.shape[0]}")
        print_landmarks(sequence[:2], idx)

if __name__ == "__main__":
    main()