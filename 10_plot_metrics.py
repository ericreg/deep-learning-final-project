#!/usr/bin/env python3
"""Parse training_metrics.json and generate evaluation benchmark plots."""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def main():
    metrics_file = Path("training_metrics.json")
    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found.")
        return

    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("Error: No data found in metrics file.")
        return

    epochs_loss = [entry["epoch"] for entry in data if entry["epoch"] > 0]
    losses = [entry["avg_loss"] for entry in data if entry["epoch"] > 0]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_loss, losses, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title("Average Cross-Entropy Loss vs. Epochs", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Average Cross-Entropy Loss", fontsize=14)
    plt.xticks(epochs_loss)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("epoch_loss_graph.png", dpi=300)
    plt.close()
    print("Saved epoch_loss_graph.png")

    tasks = list(data[0]["metrics"].keys())
    table_data = {"Task": tasks}

    for entry in data:
        epoch = entry["epoch"]
        col_name = f"Epoch {epoch}"
        col_values = []
        for task in tasks:
            acc = entry["metrics"][task]["acc"]
            
            if pd.isna(acc):
                col_values.append("n/a")
            else:
                col_values.append(f"{acc:.4f}")
                
        table_data[col_name] = col_values

    df = pd.DataFrame(table_data)

    fig, ax = plt.subplots(figsize=(12, min(2 + len(tasks) * 0.5, 8))) # height bounds based on number of tasks
    ax.axis('off')
    ax.axis('tight')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
        elif col == 0: # Task Names
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f0f0f0')
        else:
            cell.set_facecolor('#ffffff')
        
    plt.title("Benchmark Accuracy per Epoch", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("benchmark_table.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved benchmark_table.png")


if __name__ == "__main__":
    main()
