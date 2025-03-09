import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
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


def calculate_N50(contigs, genome_length):
    contigs_sorted = sorted(contigs, key=len, reverse=True)  # Sort contigs by length in descending order
    total_length = sum(len(contig) for contig in contigs_sorted)
    half_length = total_length * 0.5  # 50% of the total length
    cumulative_length = 0
    for contig in contigs_sorted:
        cumulative_length += len(contig)
        if cumulative_length >= half_length:
            return len(contig)
    return 0


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
    N50_free = calculate_N50([assembled_error_free], len(genome))
    N50_prone = calculate_N50([assembled_error_prone], len(genome))
    return (acc_free, acc_prone, coverage, genome_fraction_free, genome_fraction_prone,
            error_rate_free, error_rate_prone, N50_free, N50_prone)


def generate_plots(results):
    for param, label in zip(['N', 'l', 'p', 'min_overlap'], ["N", "l", "p", "min_overlap"]):
        param_values = [x[1] for x in results if x[0] == label]
        acc_free = [x[2][0] for x in results if x[0] == label]
        acc_prone = [x[2][1] for x in results if x[0] == label]
        coverage = [x[2][2] for x in results if x[0] == label]
        genome_fraction_free = [x[2][3] for x in results if x[0] == label]
        genome_fraction_prone = [x[2][4] for x in results if x[0] == label]
        error_rate_free = [x[2][5] for x in results if x[0] == label]
        error_rate_prone = [x[2][6] for x in results if x[0] == label]
        N50_free = [x[2][7] for x in results if x[0] == label]
        N50_prone = [x[2][8] for x in results if x[0] == label]
        plt.figure()
        plt.plot(param_values, acc_free, marker='o', linestyle='-', color='green', linewidth=2.5, label="Error-Free Accuracy")
        plt.xlabel(f"{label}")
        plt.ylabel("Accuracy")
        plt.title(f"Error-Free Accuracy vs {label}")
        plt.xlim(min(param_values), max(param_values))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/accuracy_error_free_vs_{label}.png")
        plt.close()
        plt.figure()
        plt.plot(param_values, acc_prone, marker='o', linestyle='-', color='orange', linewidth=2.5, label="Error-Prone Accuracy")
        plt.xlabel(f"{label}")
        plt.ylabel("Accuracy")
        plt.title(f"Error-Prone Accuracy vs {label}")
        plt.xlim(min(param_values), max(param_values))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/accuracy_error_prone_vs_{label}.png")
        plt.close()
        plt.figure()
        plt.plot(param_values, acc_free, marker='o', linestyle='-', color='green', linewidth=2.5, label="Error-Free Accuracy")
        plt.plot(param_values, acc_prone, marker='o', linestyle='-', color='orange', linewidth=2.5, label="Error-Prone Accuracy")
        plt.xlabel(f"{label}")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs {label}")
        plt.xlim(min(param_values), max(param_values))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/accuracy_combined_vs_{label}.png")
        plt.close()
        plt.figure()
        plt.plot(param_values, coverage, marker='o', linestyle='-', color='green', linewidth=2.5, label="Coverage (Error-Free)")
        plt.xlabel(f"{label}")
        plt.ylabel("Coverage")
        plt.title(f"Coverage (Error-Free) vs {label}")
        plt.xlim(min(param_values), max(param_values))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/coverage_error_free_vs_{label}.png")
        plt.close()
        plt.figure()
        plt.plot(param_values, coverage, marker='o', linestyle='-', color='orange', linewidth=2.5, label="Coverage (Error-Prone)")
        plt.xlabel(f"{label}")
        plt.ylabel("Coverage")
        plt.title(f"Coverage (Error-Prone) vs {label}")
        plt.xlim(min(param_values), max(param_values))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/coverage_error_prone_vs_{label}.png")
        plt.close()
        plt.figure()
        plt.plot(param_values, coverage, marker='o', linestyle='-', color='green', linewidth=2.5, label="Coverage (Error-Free)")
        plt.plot(param_values, coverage, marker='o', linestyle='-', color='orange', linewidth=2.5, label="Coverage (Error-Prone)")
        plt.xlabel(f"{label}")
        plt.ylabel("Coverage")
        plt.title(f"Coverage vs {label}")
        plt.xlim(min(param_values), max(param_values))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/coverage_combined_vs_{label}.png")
        plt.close()
        plt.figure()
        plt.plot(param_values, genome_fraction_free, marker='o', linestyle='-', color='green', linewidth=2.5, label="Genome Fraction (Error-Free)")
        plt.xlabel(f"{label}")
        plt.ylabel("Genome Fraction")
        plt.title(f"Genome Fraction (Error-Free) vs {label}")
        plt.xlim(min(param_values), max(param_values))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/genome_fraction_error_free_vs_{label}.png")
        plt.close()
        plt.figure()
        plt.plot(param_values, genome_fraction_prone, marker='o', linestyle='-', color='orange', linewidth=2.5, label="Genome Fraction (Error-Prone)")
        plt.xlabel(f"{label}")
        plt.ylabel("Genome Fraction")
        plt.title(f"Genome Fraction (Error-Prone) vs {label}")
        plt.xlim(min(param_values), max(param_values))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/genome_fraction_error_prone_vs_{label}.png")
        plt.close()
        plt.figure()
        plt.plot(param_values, genome_fraction_free, marker='o', linestyle='-', color='green', linewidth=2.5, label="Genome Fraction (Error-Free)")
        plt.plot(param_values, genome_fraction_prone, marker='o', linestyle='-', color='orange', linewidth=2.5, label="Genome Fraction (Error-Prone)")
        plt.xlabel(f"{label}")
        plt.ylabel("Genome Fraction")
        plt.title(f"Genome Fraction vs {label}")
        plt.xlim(min(param_values), max(param_values))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/genome_fraction_combined_vs_{label}.png")
        plt.close()
        plt.figure()
        plt.plot(param_values, error_rate_free, marker='o', linestyle='-', color='green', linewidth=2.5, label="Error Rate (Error-Free)")
        plt.xlabel(f"{label}")
        plt.ylabel("Error Rate")
        plt.title(f"Error Rate (Error-Free) vs {label}")
        plt.xlim(min(param_values), max(param_values))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/error_rate_error_free_vs_{label}.png")
        plt.close()
        plt.figure()
        plt.plot(param_values, error_rate_prone, marker='o', linestyle='-', color='orange', linewidth=2.5, label="Error Rate (Error-Prone)")
        plt.xlabel(f"{label}")
        plt.ylabel("Error Rate")
        plt.title(f"Error Rate (Error-Prone) vs {label}")
        plt.xlim(min(param_values), max(param_values))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/error_rate_error_prone_vs_{label}.png")
        plt.close()
        plt.figure()
        plt.plot(param_values, error_rate_free, marker='o', linestyle='-', color='green', linewidth=2.5, label="Error Rate (Error-Free)")
        plt.plot(param_values, error_rate_prone, marker='o', linestyle='-', color='orange', linewidth=2.5, label="Error Rate (Error-Prone)")
        plt.xlabel(f"{label}")
        plt.ylabel("Error Rate")
        plt.title(f"Error Rate vs {label}")
        plt.xlim(min(param_values), max(param_values))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/error_rate_combined_vs_{label}.png")
        plt.close()
        plt.figure()
        plt.plot(param_values, N50_free, marker='o', linestyle='-', color='green', linewidth=2.5, label="N50 (Error-Free)")
        plt.xlabel(f"{label}")
        plt.ylabel("N50")
        plt.title(f"N50 (Error-Free) vs {label}")
        plt.xlim(min(param_values), max(param_values))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/N50_error_free_vs_{label}.png")
        plt.close()
        plt.figure()
        plt.plot(param_values, N50_prone, marker='o', linestyle='-', color='orange', linewidth=2.5, label="N50 (Error-Prone)")
        plt.xlabel(f"{label}")
        plt.ylabel("N50")
        plt.title(f"N50 (Error-Prone) vs {label}")
        plt.xlim(min(param_values), max(param_values))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/N50_error_prone_vs_{label}.png")
        plt.close()
        plt.figure()
        plt.plot(param_values, N50_free, marker='o', linestyle='-', color='green', linewidth=2.5, label="N50 (Error-Free)")
        plt.plot(param_values, N50_prone, marker='o', linestyle='-', color='orange', linewidth=2.5, label="N50 (Error-Prone)")
        plt.xlabel(f"{label}")
        plt.ylabel("N50")
        plt.title(f"N50 vs {label}")
        plt.xlim(min(param_values), max(param_values))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/N50_combined_vs_{label}.png")
        plt.close()
    print("All basic plots saved in the 'plots' folder!")


def generate_3d_heatmap(genome, Ns, ls, ps, min_overlaps):
    N_values_3d = []
    l_values_3d = []
    p_values_3d = []
    min_overlap_values_3d = []
    error_rate_values_3d = []
    coverage_values_3d = []
    genome_fraction_values_3d = []
    N50_values_3d = []
    min_overlap_values_heatmap = sorted(min_overlaps)
    N_values_heatmap = sorted(Ns)
    error_rate_grid = np.zeros((len(min_overlap_values_heatmap), len(N_values_heatmap)))
    for N in Ns:
        for l in ls:
            for p in ps:
                for min_overlap in min_overlaps:
                    result = run_experiment(genome, N, l, p, min_overlap)
                    acc_free, acc_prone, coverage, genome_fraction_free, genome_fraction_prone, error_rate_free, error_rate_prone, N50_free, N50_prone = result
                    N_values_3d.append(N)
                    l_values_3d.append(l)
                    p_values_3d.append(p)
                    min_overlap_values_3d.append(min_overlap)
                    error_rate_values_3d.append(error_rate_prone)
                    coverage_values_3d.append(coverage)
                    genome_fraction_values_3d.append(genome_fraction_prone)
                    N50_values_3d.append(N50_prone)
                    min_overlap_index = min_overlap_values_heatmap.index(min_overlap)
                    N_index = N_values_heatmap.index(N)
                    error_rate_grid[min_overlap_index, N_index] = error_rate_prone
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(N_values_3d, l_values_3d, error_rate_values_3d, c=error_rate_values_3d, cmap='inferno', marker='o', s=100, depthshade=True)
    ax.set_xlabel("Number of Reads (N)", fontsize=12)
    ax.set_ylabel("Read Length (l)", fontsize=12)
    ax.set_zlabel("Error Rate", fontsize=12)
    plt.title("3D Plot: N, l, and Error Rate", fontsize=16)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Error Rate', fontsize=12)
    plt.savefig("plots/3d_plot_N_l_error_rate_all_combinations.png", bbox_inches='tight', transparent=True)
    plt.close()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(p_values_3d, N_values_3d, error_rate_values_3d, c=error_rate_values_3d, cmap='viridis', marker='o', s=100, depthshade=True)
    ax.set_xlabel("Error Probability (p)", fontsize=12)
    ax.set_ylabel("Number of Reads (N)", fontsize=12)
    ax.set_zlabel("Error Rate", fontsize=12)
    plt.title("3D Plot: p, N, and Error Rate", fontsize=16)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Error Rate', fontsize=12)
    plt.savefig("plots/3d_plot_p_N_error_rate_all_combinations.png", bbox_inches='tight', transparent=True)
    plt.close()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(p_values_3d, l_values_3d, error_rate_values_3d, c=error_rate_values_3d, cmap='plasma', marker='o', s=100, depthshade=True)
    ax.set_xlabel("Error Probability (p)", fontsize=12)
    ax.set_ylabel("Read Length (l)", fontsize=12)
    ax.set_zlabel("Error Rate", fontsize=12)
    plt.title("3D Plot: p, l, and Error Rate", fontsize=16)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Error Rate', fontsize=12)
    plt.savefig("plots/3d_plot_p_l_error_rate_all_combinations.png", bbox_inches='tight', transparent=True)
    plt.close()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(N_values_3d, p_values_3d, coverage_values_3d, c=coverage_values_3d, cmap='cool', marker='o', s=100, depthshade=True)
    ax.set_xlabel("Number of Reads (N)", fontsize=12)
    ax.set_ylabel("Error Probability (p)", fontsize=12)
    ax.set_zlabel("Coverage", fontsize=12)
    plt.title("3D Plot: N, p, and Coverage", fontsize=16)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Coverage', fontsize=12)
    plt.savefig("plots/3d_plot_N_p_coverage_all_combinations.png", bbox_inches='tight', transparent=True)
    plt.close()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(l_values_3d, min_overlap_values_3d, genome_fraction_values_3d, c=genome_fraction_values_3d, cmap='spring', marker='o', s=100, depthshade=True)
    ax.set_xlabel("Read Length (l)", fontsize=12)
    ax.set_ylabel("Minimum Overlap (min_overlap)", fontsize=12)
    ax.set_zlabel("Genome Fraction", fontsize=12)
    plt.title("3D Plot: l, min_overlap, and Genome Fraction", fontsize=16)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Genome Fraction', fontsize=12)
    plt.savefig("plots/3d_plot_l_min_overlap_genome_fraction_all_combinations.png", bbox_inches='tight', transparent=True)
    plt.close()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(N_values_3d, l_values_3d, N50_values_3d, c=N50_values_3d, cmap='winter', marker='o', s=100, depthshade=True)
    ax.set_xlabel("Number of Reads (N)", fontsize=12)
    ax.set_ylabel("Read Length (l)", fontsize=12)
    ax.set_zlabel("N50", fontsize=12)
    plt.title("3D Plot: N, l, and N50", fontsize=16)
    cbar = plt.colorbar(scatter)
    cbar.set_label('N50', fontsize=12)
    plt.savefig("plots/3d_plot_N_l_N50_all_combinations.png", bbox_inches='tight', transparent=True)
    plt.close()
    plt.figure(figsize=(10, 8))
    sns.heatmap(error_rate_grid, annot=True, fmt=".4f", xticklabels=N_values_heatmap, yticklabels=min_overlap_values_heatmap, cmap="YlOrRd", cbar_kws={'label': 'Error Rate'})
    plt.xlabel("Number of Reads (N)", fontsize=14)
    plt.ylabel("Minimum Overlap (min_overlap)", fontsize=14)
    plt.title("Error Rate Heatmap (min_overlap vs N)", fontsize=16)
    plt.savefig("plots/error_rate_heatmap_min_overlap_vs_N_all_combinations.png", bbox_inches='tight', transparent=True)
    plt.close()
    print("All 3D plots and heatmap generated!")


def run_experiment_wrapper(args):
    return run_experiment(*args)


def run_full_experiments(genome):
    Ns = [100, 500, 2000, 5000, 10000]
    ls = [25, 50, 100, 200, 300, 400]
    ps = [0.005, 0.01, 0.05, 0.1]
    min_overlaps = [5, 10, 30, 50]

    params = []
    for N in Ns:
        params.append(('N', (genome, N, 100, 0.01, 20)))
    for l in ls:
        params.append(('l', (genome, 1000, l, 0.01, 20)))
    for p in ps:
        params.append(('p', (genome, 1000, 100, p, 20)))
    for min_overlap in min_overlaps:
        params.append(('min_overlap', (genome, 1000, 100, 0.01, min_overlap)))

    with Pool(processes=7) as pool:
        results = pool.map(run_experiment_wrapper, [x[1] for x in params])

    results = [(params[i][0], params[i][1][1], res) for i, res in enumerate(results)]
    generate_plots(results)
    generate_3d_heatmap(genome, Ns, ls, ps, min_overlaps)


if __name__ == '__main__':
    fasta_file = "sequence.fasta"
    genome = sequence(fasta_file)
    run_full_experiments(genome)
