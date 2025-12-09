import csv
import matplotlib.pyplot as plt

xs = []
ys_brute = []
ys_z3 = []

with open("benchmark_results.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        xs.append(int(row["num_vars"]))
        ys_brute.append(float(row["avg_bruteforce_ms"]))
        ys_z3.append(float(row["avg_z3_ms"]))

plt.figure()
plt.plot(xs, ys_brute, marker="o", label="Брутфорс")
plt.plot(xs, ys_z3, marker="o", label="Z3 (митор)")
plt.xlabel("Число входных переменных")
plt.ylabel("Среднее время, мс")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("time_vs_vars.png", dpi=300)
plt.show()
