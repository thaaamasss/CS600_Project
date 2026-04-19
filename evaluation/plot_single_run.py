import matplotlib.pyplot as plt
import numpy as np
import os

def generate_run_plots(dataset_name, model_name, exec_time, pre_target, pre_retain, post_target, post_retain):
    """
    Generates and saves accuracy and time plots for a single SSN unlearning run.
    """
    # Create directory if it doesn't exist
    save_dir = "results/plots/ssn_results"
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # PLOT 1: ACCURACY (PRE VS POST SSN)
    # ---------------------------------------------------------
    categories = ['Target Class (Deleted)', 'Retained Classes']
    pre_acc = [pre_target, pre_retain]
    post_acc = [post_target, post_retain]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, pre_acc, width, label='Pre-SSN (Baseline)', color='#34495E')
    rects2 = ax.bar(x + width/2, post_acc, width, label='Post-SSN (Necrotic)', color='#E74C3C')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'SSN Impact: {dataset_name} ({model_name} model)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 110)

    # Add data labels
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    acc_filename = f"{dataset_name}_{model_name}_SSN_Accuracy.png"
    acc_save_path = os.path.join(save_dir, acc_filename)
    plt.savefig(acc_save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Accuracy Plot: {acc_save_path}")

    # ---------------------------------------------------------
    # PLOT 2: EXECUTION TIME
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    bar = ax.bar(['SSN Execution'], [exec_time], color='#3498DB', width=0.4)
    
    ax.set_ylabel('Time (Seconds)', fontsize=12)
    ax.set_title(f'SSN Unlearning Time: {dataset_name} ({model_name})', fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add time label
    yval = bar[0].get_height()
    ax.annotate(f'{yval:.4f} s',
                xy=(bar[0].get_x() + bar[0].get_width() / 2, yval),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Scale Y axis slightly above the time for visual padding
    ax.set_ylim(0, exec_time + (exec_time * 0.2) + 1)

    time_filename = f"{dataset_name}_{model_name}_SSN_Time.png"
    time_save_path = os.path.join(save_dir, time_filename)
    plt.savefig(time_save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Execution Time Plot: {time_save_path}")