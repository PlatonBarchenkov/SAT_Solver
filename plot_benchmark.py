import csv
import matplotlib.pyplot as plt


def plot_benchmark():
    xs = []
    ys_brute = []
    ys_dpll = []
    ys_cdcl = []
    ys_z3 = []

    with open('benchmark_results.csv', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['n'])
            xs.append(n)

            if row['brute']:
                ys_brute.append(float(row['brute']))
            else:
                ys_brute.append(None)

            ys_dpll.append(float(row['dpll']))
            ys_cdcl.append(float(row['cdcl']))
            ys_z3.append(float(row['z3']))

    plt.figure(figsize=(10, 6))

    xs_brute = [x for x, y in zip(xs, ys_brute) if y is not None]
    ys_brute_clean = [y for y in ys_brute if y is not None]

    plt.plot(xs_brute, ys_brute_clean, marker='o', markersize=4, linewidth=1.5, label='Brute Force', color='tab:red')
    plt.plot(xs, ys_dpll, marker='o', markersize=4, linewidth=1.5, label='DPLL', color='tab:blue')
    plt.plot(xs, ys_cdcl, marker='o', markersize=4, linewidth=1.5, label='CDCL', color='tab:green')
    plt.plot(xs, ys_z3, marker='o', markersize=4, linewidth=1.5, label='Z3', color='tab:purple')

    plt.xlabel('Number of Variables (N)', fontsize=12)
    plt.ylabel('Time (ms) - Log Scale', fontsize=12)
    plt.title('SAT Solvers Performance Comparison', fontsize=14)

    plt.yscale('log')

    plt.grid(True, which="major", ls="-", alpha=0.3)
    plt.grid(True, which="minor", ls=":", alpha=0.1)

    plt.legend(fontsize=10)
    plt.tight_layout()

    plt.savefig('benchmark_plot.png', dpi=300)
    print("Graph saved to benchmark_plot.png")
    plt.show()


if __name__ == "__main__":
    plot_benchmark()
