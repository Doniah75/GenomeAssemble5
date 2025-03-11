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


def run_experiment_wrapper(args):
    return run_experiment(*args)


def generate_3d_heatmap(genome, Ns, ls, ps, min_overlaps, output_folder):
    # Lists to store results for error-free and error-prone scenarios
    N_values_3d_free, l_values_3d_free, p_values_3d_free, min_overlap_values_3d_free = [], [], [], []
    error_rate_values_3d_free, coverage_values_3d_free, genome_fraction_values_3d_free, N50_values_3d_free = [], [], [], []

    N_values_3d_prone, l_values_3d_prone, p_values_3d_prone, min_overlap_values_3d_prone = [], [], [], []
    error_rate_values_3d_prone, coverage_values_3d_prone, genome_fraction_values_3d_prone, N50_values_3d_prone = [], [], [], []

    min_overlap_values_heatmap = sorted(min_overlaps)
    N_values_heatmap = sorted(Ns)
    error_rate_grid_free = np.zeros((len(min_overlap_values_heatmap), len(N_values_heatmap)))
    error_rate_grid_prone = np.zeros((len(min_overlap_values_heatmap), len(N_values_heatmap)))

    # Create a list of all parameter combinations
    param_combinations = [(genome, N, l, p, min_overlap) for N in Ns for l in ls for p in ps for min_overlap in min_overlaps]

    # Use multiprocessing to compute results in parallel
    with Pool(processes=7) as pool:
        results = pool.map(run_experiment_wrapper, param_combinations)

    # Collect results for error-free and error-prone scenarios
    for i, result in enumerate(results):
        N, l, p, min_overlap = param_combinations[i][1], param_combinations[i][2], param_combinations[i][3], param_combinations[i][4]
        acc_free, acc_prone, coverage, genome_fraction_free, genome_fraction_prone, error_rate_free, error_rate_prone, N50_free, N50_prone = result

        # Error-free results
        N_values_3d_free.append(N)
        l_values_3d_free.append(l)
        p_values_3d_free.append(p)
        min_overlap_values_3d_free.append(min_overlap)
        error_rate_values_3d_free.append(error_rate_free)
        coverage_values_3d_free.append(coverage)
        genome_fraction_values_3d_free.append(genome_fraction_free)
        N50_values_3d_free.append(N50_free)
        min_overlap_index = min_overlap_values_heatmap.index(min_overlap)
        N_index = N_values_heatmap.index(N)
        error_rate_grid_free[min_overlap_index, N_index] = error_rate_free

        # Error-prone results
        N_values_3d_prone.append(N)
        l_values_3d_prone.append(l)
        p_values_3d_prone.append(p)
        min_overlap_values_3d_prone.append(min_overlap)
        error_rate_values_3d_prone.append(error_rate_prone)
        coverage_values_3d_prone.append(coverage)
        genome_fraction_values_3d_prone.append(genome_fraction_prone)
        N50_values_3d_prone.append(N50_prone)
        error_rate_grid_prone[min_overlap_index, N_index] = error_rate_prone

    # Generate 3D plots and heatmaps for error-free scenarios
    generate_plots(N_values_3d_free, l_values_3d_free, p_values_3d_free, min_overlap_values_3d_free,
                   error_rate_values_3d_free, coverage_values_3d_free, genome_fraction_values_3d_free, N50_values_3d_free,
                   error_rate_grid_free, min_overlap_values_heatmap, N_values_heatmap, "error_free", output_folder)

    # Generate 3D plots and heatmaps for error-prone scenarios
    generate_plots(N_values_3d_prone, l_values_3d_prone, p_values_3d_prone, min_overlap_values_3d_prone,
                   error_rate_values_3d_prone, coverage_values_3d_prone, genome_fraction_values_3d_prone, N50_values_3d_prone,
                   error_rate_grid_prone, min_overlap_values_heatmap, N_values_heatmap, "error_prone", output_folder)

    print(f"All 3D plots and heatmaps generated for {output_folder}!")


def generate_plots(N_values, l_values, p_values, min_overlap_values, error_rate_values, coverage_values,
                   genome_fraction_values, N50_values, error_rate_grid, min_overlap_values_heatmap, N_values_heatmap, suffix, output_folder):
    # Generate 3D plots
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(N_values, l_values, error_rate_values, c=error_rate_values, cmap='inferno', marker='o', s=100, depthshade=True)
    ax.set_xlabel("Number of Reads (N)", fontsize=12)
    ax.set_ylabel("Read Length (l)", fontsize=12)
    ax.set_zlabel("Error Rate", fontsize=12)
    plt.title(f"3D Plot: N, l, and Error Rate ({suffix})", fontsize=16)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Error Rate', fontsize=12)
    plt.savefig(f"{output_folder}/3d_plot_N_l_error_rate_{suffix}.png", bbox_inches='tight', transparent=True)
    plt.close()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(p_values, N_values, error_rate_values, c=error_rate_values, cmap='viridis', marker='o', s=100, depthshade=True)
    ax.set_xlabel("Error Probability (p)", fontsize=12)
    ax.set_ylabel("Number of Reads (N)", fontsize=12)
    ax.set_zlabel("Error Rate", fontsize=12)
    plt.title(f"3D Plot: p, N, and Error Rate ({suffix})", fontsize=16)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Error Rate', fontsize=12)
    plt.savefig(f"{output_folder}/3d_plot_p_N_error_rate_{suffix}.png", bbox_inches='tight', transparent=True)
    plt.close()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(p_values, l_values, error_rate_values, c=error_rate_values, cmap='plasma', marker='o', s=100, depthshade=True)
    ax.set_xlabel("Error Probability (p)", fontsize=12)
    ax.set_ylabel("Read Length (l)", fontsize=12)
    ax.set_zlabel("Error Rate", fontsize=12)
    plt.title(f"3D Plot: p, l, and Error Rate ({suffix})", fontsize=16)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Error Rate', fontsize=12)
    plt.savefig(f"{output_folder}/3d_plot_p_l_error_rate_{suffix}.png", bbox_inches='tight', transparent=True)
    plt.close()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(N_values, p_values, coverage_values, c=coverage_values, cmap='cool', marker='o', s=100, depthshade=True)
    ax.set_xlabel("Number of Reads (N)", fontsize=12)
    ax.set_ylabel("Error Probability (p)", fontsize=12)
    ax.set_zlabel("Coverage", fontsize=12)
    plt.title(f"3D Plot: N, p, and Coverage ({suffix})", fontsize=16)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Coverage', fontsize=12)
    plt.savefig(f"{output_folder}/3d_plot_N_p_coverage_{suffix}.png", bbox_inches='tight', transparent=True)
    plt.close()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(l_values, min_overlap_values, genome_fraction_values, c=genome_fraction_values, cmap='spring', marker='o', s=100, depthshade=True)
    ax.set_xlabel("Read Length (l)", fontsize=12)
    ax.set_ylabel("Minimum Overlap (min_overlap)", fontsize=12)
    ax.set_zlabel("Genome Fraction", fontsize=12)
    plt.title(f"3D Plot: l, min_overlap, and Genome Fraction ({suffix})", fontsize=16)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Genome Fraction', fontsize=12)
    plt.savefig(f"{output_folder}/3d_plot_l_min_overlap_genome_fraction_{suffix}.png", bbox_inches='tight', transparent=True)
    plt.close()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(N_values, l_values, N50_values, c=N50_values, cmap='winter', marker='o', s=100, depthshade=True)
    ax.set_xlabel("Number of Reads (N)", fontsize=12)
    ax.set_ylabel("Read Length (l)", fontsize=12)
    ax.set_zlabel("N50", fontsize=12)
    plt.title(f"3D Plot: N, l, and N50 ({suffix})", fontsize=16)
    cbar = plt.colorbar(scatter)
    cbar.set_label('N50', fontsize=12)
    plt.savefig(f"{output_folder}/3d_plot_N_l_N50_{suffix}.png", bbox_inches='tight', transparent=True)
    plt.close()

    # Generate heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(error_rate_grid, annot=True, fmt=".4f", xticklabels=N_values_heatmap, yticklabels=min_overlap_values_heatmap, cmap="YlOrRd", cbar_kws={'label': 'Error Rate'})
    plt.xlabel("Number of Reads (N)", fontsize=14)
    plt.ylabel("Minimum Overlap (min_overlap)", fontsize=14)
    plt.title(f"Error Rate Heatmap (min_overlap vs N) ({suffix})", fontsize=16)
    plt.savefig(f"{output_folder}/error_rate_heatmap_min_overlap_vs_N_{suffix}.png", bbox_inches='tight', transparent=True)
    plt.close()


def run_full_experiments(genome, fraction, output_folder):
    Ns = [100, 500, 2000, 5000, 10000]
    ls = [25, 50, 100, 200, 300, 400]
    ps = [0.005, 0.01, 0.05, 0.1]
    min_overlaps = [5, 10, 30, 50]

    # Extract the fraction of the genome
    genome_length = len(genome)
    fraction_length = int(genome_length * fraction)
    genome_fraction = genome[:fraction_length]

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    generate_3d_heatmap(genome_fraction, Ns, ls, ps, min_overlaps, output_folder)


if __name__ == '__main__':
    fasta_file = "sequence.fasta"
    genome = sequence(fasta_file)
    run_full_experiments(genome, 0.25, "plots1")
    run_full_experiments(genome, 0.5, "plots2")
    run_full_experiments(genome, 0.75, "plots3")
