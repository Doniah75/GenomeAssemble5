import os
import random
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from multiprocessing import Pool

if not os.path.exists("plots"):
    os.makedirs("plots")


def sequence(fasta_file):
    with open(fasta_file, 'r') as file:
        lines = file.readlines()
    genome = ''.join(line.strip() for line in lines if not line.startswith('>'))
    print(f"Extracted genome of length {len(genome)}")
    return genome


def generate_reads(genome, N, l):
    genome_length = len(genome)
    max_possible_reads = max(1, genome_length - l + 1)
    N = min(N, max_possible_reads)
    reads = [genome[i:i + l] for i in sorted(random.sample(range(max_possible_reads), N))]
    return reads


def introduce_errors(reads, p):
    bases = ['A', 'T', 'C', 'G']

    def mutate(base):
        return random.choice([b for b in bases if b != base]) if random.random() < p else base

    return [''.join(mutate(base) for base in read) for read in reads]


def compute_overlap(s1, s2, min_overlap=3):
    max_len = min(len(s1), len(s2))
    for i in range(max_len, min_overlap - 1, -1):
        if s1[-i:] == s2[:i]:
            return i
    return 0


def build_overlap_graph(reads, min_overlap=3):
    graph = defaultdict(dict)
    for i, read1 in enumerate(reads):
        for j, read2 in enumerate(reads):
            if i != j:
                overlap = compute_overlap(read1, read2, min_overlap)
                if overlap > 0:
                    graph[read1][read2] = overlap
    return graph


def assemble_genome(reads, min_overlap=3):
    graph = build_overlap_graph(reads, min_overlap)
    assembled = reads[0]
    used_reads = {assembled}
    while len(used_reads) < len(reads):
        best_read, best_overlap = None, 0
        for read in graph[assembled]:
            if read not in used_reads and graph[assembled][read] > best_overlap:
                best_read, best_overlap = read, graph[assembled][read]
        if best_read:
            assembled += best_read[best_overlap:]
            used_reads.add(best_read)
        else:
            break
    return assembled


def calculate_accuracy(genome, assembled):
    min_len = min(len(genome), len(assembled))
    return sum(1 for a, b in zip(genome[:min_len], assembled[:min_len]) if a == b) / min_len


def calculate_coverage(reads, genome_length):
    total_bases = sum(len(read) for read in reads)
    return total_bases / genome_length


def calculate_genome_fraction(contigs, genome_length):
    total_contig_length = sum(len(contig) for contig in contigs)
    return total_contig_length / genome_length


def calculate_error_rate(assembled_genome, original_genome):
    min_len = min(len(assembled_genome), len(original_genome))
    mismatches = sum(1 for a, b in zip(assembled_genome[:min_len], original_genome[:min_len]) if a != b)
    return mismatches / min_len


def run_experiment(genome, N, l, p, min_overlap):
    print(f"Running: N={N}, l={l}, p={p}, min_overlap={min_overlap}")
    reads = generate_reads(genome, N, l)
    error_reads = introduce_errors(reads, p)

    assembled_error_free = assemble_genome(reads, min_overlap)
    assembled_error_prone = assemble_genome(error_reads, min_overlap)

    acc_free = calculate_accuracy(genome, assembled_error_free)
    acc_prone = calculate_accuracy(genome, assembled_error_prone)
    coverage = calculate_coverage(reads, len(genome))
    genome_fraction_free = calculate_genome_fraction([assembled_error_free], len(genome))
    genome_fraction_prone = calculate_genome_fraction([assembled_error_prone], len(genome))
    error_rate_free = calculate_error_rate(assembled_error_free, genome)
    error_rate_prone = calculate_error_rate(assembled_error_prone, genome)

    return (acc_free, acc_prone, coverage, genome_fraction_free, genome_fraction_prone,
            error_rate_free, error_rate_prone)


def generate_plots(results):
    param = 'p'
    label = "p"
    param_values = [x[1] for x in results if x[0] == param]

    if not param_values:
        print(f"Skipping plots for {label} as there are no valid data points.")
        return

    acc_free = [x[2][0] for x in results]
    acc_prone = [x[2][1] for x in results]
    coverage = [x[2][2] for x in results]
    genome_fraction_free = [x[2][3] for x in results]
    genome_fraction_prone = [x[2][4] for x in results]
    error_rate_free = [x[2][5] for x in results]
    error_rate_prone = [x[2][6] for x in results]

    metrics = {
        "Accuracy": (acc_free, acc_prone),
        "Coverage": (coverage, None),
        "Genome Fraction": (genome_fraction_free, genome_fraction_prone),
        "Error Rate": (error_rate_free, error_rate_prone),
    }

    colors = {"Error-Free": "green", "Error-Prone": "orange"}

    for metric_name, (free_values, prone_values) in metrics.items():
        plt.figure()
        if free_values:
            plt.plot(param_values, free_values, marker='o', linestyle='-', linewidth=2.5,
                     color=colors["Error-Free"], label=f"{metric_name} (Error-Free)")
        if prone_values:
            plt.plot(param_values, prone_values, marker='o', linestyle='-', linewidth=2.5,
                     color=colors["Error-Prone"], label=f"{metric_name} (Error-Prone)")
        plt.xlabel(label)
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs {label}")
        plt.xlim(min(param_values), max(param_values))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/{metric_name.replace(' ', '_').lower()}_vs_{label}.png")
        plt.close()

    print("All plots saved successfully in the 'plots' folder!")


def run_experiment_wrapper(args):
    return run_experiment(*args)


def run_full_experiments(genome):
    ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    params = [('p', (genome, 1000, 100, p, 20)) for p in ps]

    with Pool(processes=7) as pool:
        results = pool.map(run_experiment_wrapper, [x[1] for x in params])

    results = [(params[i][0], params[i][1][3], res) for i, res in enumerate(results)]
    generate_plots(results)


if __name__ == '__main__':
    fasta_file = "sequence.fasta"
    genome = sequence(fasta_file)
    run_full_experiments(genome)
