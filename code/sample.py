from Bio import SeqIO
from random import shuffle

def random_sample_data(virus, num_seqs):
    '''
    Helper function. Do not call directly.

    Reads and parses a FASTA file. Randomly samples sequences from the file and
    writes to a new FASTA file.

    :param file_name: path to .fasta file
    :param num_seqs: number of sequences to randomly sample

    :return: none
    '''
    with open(f"fasta_data_new/{virus}.fasta") as file:
        parsed = SeqIO.parse(file,'fasta')
        seqs = list(parsed)

    shuffle(seqs)

    with open(f"fasta_samples/{virus}.fasta", "w") as file:
        SeqIO.write(seqs[:num_seqs], file, "fasta")
    
num_seqs = 1000
viruses = ["dengue", "hepatitis", "influenza", "mers", "sarscov1", "sarscov2"]

for virus in viruses:
    random_sample_data(virus, num_seqs)