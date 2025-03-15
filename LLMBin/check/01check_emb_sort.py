import numpy as np
print("*****************逆序排列***************")
dna_sequences = ['a', 'ab', 'abcd', 'abc']
lengths = [len(seq) for seq in dna_sequences]
print(f"lengths: {lengths}")
idx = np.argsort(lengths)[::-1]
dna_sequences = [dna_sequences[i] for i in idx]
print(f"idx: {idx}, dna_sequences: {dna_sequences}")

emb = np.array([4,3,2,1])
print(f"emb: {emb}")
new_emb = emb[np.argsort(idx)]
print(f"np.argsort(idx): {np.argsort(idx)}")
print(f"new_emb: {new_emb}")


print("*****************顺序排列******************")
dna_sequences = ['a', 'ab', 'abcd', 'abc']
lengths = [len(seq) for seq in dna_sequences]
print(f"lengths: {lengths}")
idx = np.argsort(lengths)
dna_sequences = [dna_sequences[i] for i in idx]
print(f"idx: {idx}, dna_sequences: {dna_sequences}")

emb = np.array([1,2,3,4])
print(f"emb: {emb}")
new_emb = emb[np.argsort(idx)]
print(f"np.argsort(idx): {np.argsort(idx)}")
print(f"new_emb: {new_emb}")