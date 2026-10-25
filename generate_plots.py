import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set global style for presentation slides
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk") # Makes text larger for presentations

# Professional, accessible colors
COLOR_BASELINE = '#A0AEC0' # Muted gray-blue
COLOR_ENHANCED = '#DD6B20' # Professional orange
COLOR_TEXT = '#2D3748'

# ─────────────────────────────────────────────
#  RAW RESULTS DATA
# ─────────────────────────────────────────────

sst2_baseline = {
    "accuracy":  0.8778,
    "macro_f1":  0.88,
    "classes":   ["Negative", "Positive"],
    "precision": [0.85, 0.91],
    "recall":    [0.89, 0.87],
    "f1":        [0.87, 0.89],
    "support":   [4532, 5702],
}
sst2_enhanced = {
    "accuracy":  0.8935,
    "macro_f1":  0.89,
    "classes":   ["Negative", "Positive"],
    "precision": [0.88, 0.90],
    "recall":    [0.88, 0.91],
    "f1":        [0.88, 0.90],
    "support":   [4532, 5702],
}

emo_classes   = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
emo_baseline  = {
    "accuracy":  0.8933,
    "macro_f1":  0.86,
    "classes":   emo_classes,
    "precision": [0.95, 0.92, 0.72, 0.90, 0.87, 0.73],
    "recall":    [0.93, 0.89, 0.86, 0.91, 0.84, 0.86],
    "f1":        [0.94, 0.91, 0.78, 0.90, 0.86, 0.79],
    "support":   [700, 804, 196, 324, 290, 86],
}
emo_enhanced  = {
    "accuracy":  0.8775,
    "macro_f1":  0.84,
    "classes":   emo_classes,
    "precision": [0.93, 0.91, 0.70, 0.88, 0.86, 0.69],
    "recall":    [0.92, 0.88, 0.82, 0.89, 0.83, 0.84],
    "f1":        [0.92, 0.89, 0.76, 0.89, 0.84, 0.75],
    "support":   [700, 804, 196, 324, 290, 86],
}

# Ensure output directory exists
os.makedirs("plots", exist_ok=True)

def plot_single_dataset_metrics(dataset_name, base_acc, enh_acc, base_f1, enh_f1, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metrics = ['Accuracy', 'Macro F1']
    x = np.arange(len(metrics))
    width = 0.35
    
    base_vals = [base_acc, base_f1]
    enh_vals = [enh_acc, enh_f1]
    
    rects1 = ax.bar(x - width/2, base_vals, width, label='Baseline TF-IDF', color=COLOR_BASELINE, edgecolor='white')
    rects2 = ax.bar(x + width/2, enh_vals, width, label='Enhanced Pipeline', color=COLOR_ENHANCED, edgecolor='white')
    
    ax.set_ylabel('Score', fontsize=14, color=COLOR_TEXT)
    ax.set_title(f'{dataset_name} Overall Performance', fontsize=16, fontweight='bold', color=COLOR_TEXT, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=14, color=COLOR_TEXT)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    
    for i, rect in enumerate(rects1):
        height = rect.get_height()
        val_str = f'{height:.4f}' if i == 0 else f'{height:.2f}'
        ax.annotate(val_str, xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color=COLOR_TEXT)
                    
    for i, rect in enumerate(rects2):
        height = rect.get_height()
        val_str = f'{height:.4f}' if i == 0 else f'{height:.2f}'
        ax.annotate(val_str, xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color=COLOR_TEXT)

    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig(f'plots/{filename}', dpi=300, transparent=False, bbox_inches='tight')
    plt.close()

def plot_both_datasets_metrics():
    datasets = ['SST-2\n(Binary)', 'Emotion\n(6-Class)']
    x = np.arange(len(datasets))
    width = 0.35
    
    base_acc = [sst2_baseline['accuracy'], emo_baseline['accuracy']]
    enh_acc = [sst2_enhanced['accuracy'], emo_enhanced['accuracy']]
    
    base_f1 = [sst2_baseline['macro_f1'], emo_baseline['macro_f1']]
    enh_f1 = [sst2_enhanced['macro_f1'], emo_enhanced['macro_f1']]

    # Plot Accuracy and F1 side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Accuracy Plot
    ax1 = axes[0]
    rects1 = ax1.bar(x - width/2, base_acc, width, label='Baseline TF-IDF', color=COLOR_BASELINE, edgecolor='white')
    rects2 = ax1.bar(x + width/2, enh_acc, width, label='Enhanced Pipeline', color=COLOR_ENHANCED, edgecolor='white')
    
    ax1.set_ylabel('Accuracy', fontsize=14, color=COLOR_TEXT)
    ax1.set_title('Overall Accuracy Comparison', fontsize=16, fontweight='bold', color=COLOR_TEXT, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=14, color=COLOR_TEXT)
    ax1.set_ylim(0, 1.05)
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax1.annotate(f'{height:.2%}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color=COLOR_TEXT)
                    
    # 2. Macro F1 Plot
    ax2 = axes[1]
    rects1 = ax2.bar(x - width/2, base_f1, width, label='Baseline TF-IDF', color=COLOR_BASELINE, edgecolor='white')
    rects2 = ax2.bar(x + width/2, enh_f1, width, label='Enhanced Pipeline', color=COLOR_ENHANCED, edgecolor='white')
    
    ax2.set_ylabel('Macro F1 Score', fontsize=14, color=COLOR_TEXT)
    ax2.set_title('Macro F1 Comparison', fontsize=16, fontweight='bold', color=COLOR_TEXT, pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=14, color=COLOR_TEXT)
    ax2.set_ylim(0, 1.05)
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color=COLOR_TEXT)

    sns.despine(left=True)
    plt.tight_layout()
    
    # Single legend for the entire figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False, fontsize=14)
    
    plt.savefig('plots/both_datasets_metrics.png', dpi=300, transparent=False, bbox_inches='tight')
    plt.close()

def draw_table(title, data_rows, columns, filename):
    fig, ax = plt.subplots(figsize=(10, len(data_rows) * 0.4 + 1.5))
    ax.axis('off')
    
    table_data = []
    cell_colors = []
    
    for row in data_rows:
        metric, cls, base, enh = row
        base_val, enh_val = float(base), float(enh)
        
        base_str = f"{base_val:.4f}" if "Accuracy" in metric else f"{base_val:.2f}"
        enh_str = f"{enh_val:.4f}" if "Accuracy" in metric else f"{enh_val:.2f}"
        
        # Highlight higher score
        row_colors = ['white', 'white', 'white', 'white'] 
        if base_val > enh_val:
            row_colors[2] = '#C6F6D5' # Green highlight
            row_colors[3] = '#FED7D7' # Red lowlight
        elif enh_val > base_val:
            row_colors[2] = '#FED7D7'
            row_colors[3] = '#C6F6D5'
            
        table_data.append([metric, cls, base_str, enh_str])
        cell_colors.append(row_colors)
        
    table = ax.table(cellText=table_data, colLabels=columns, cellColours=cell_colors, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    
    # Style Header
    for j, col in enumerate(columns):
        cell = table[0, j]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2D3748')
        
    # Make "Overall" rows bold
    for i, row in enumerate(data_rows):
        if row[1] == 'Overall':
            for j in range(4):
                table[i+1, j].set_text_props(weight='bold')
                
    plt.title(title, fontsize=18, fontweight='bold', pad=20, color='#2D3748')
    plt.tight_layout()
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def generate_tables():
    columns = ['Metric', 'Class', 'Baseline TF-IDF', 'Enhanced Pipeline']
    
    # SST-2 Table
    sst2_rows = [
        ['Accuracy', 'Overall', sst2_baseline['accuracy'], sst2_enhanced['accuracy']],
        ['Macro F1', 'Overall', sst2_baseline['macro_f1'], sst2_enhanced['macro_f1']],
    ]
    for i, cls in enumerate(sst2_baseline['classes']):
        sst2_rows.append(['Precision', cls, sst2_baseline['precision'][i], sst2_enhanced['precision'][i]])
        sst2_rows.append(['Recall', cls, sst2_baseline['recall'][i], sst2_enhanced['recall'][i]])
        sst2_rows.append(['F1 Score', cls, sst2_baseline['f1'][i], sst2_enhanced['f1'][i]])
        
    draw_table("SST-2 Performance Comparison", sst2_rows, columns, "sst2_table.png")
    
    # Emotion Table
    emo_rows = [
        ['Accuracy', 'Overall', emo_baseline['accuracy'], emo_enhanced['accuracy']],
        ['Macro F1', 'Overall', emo_baseline['macro_f1'], emo_enhanced['macro_f1']],
    ]
    for i, cls in enumerate(emo_baseline['classes']):
        emo_rows.append(['Precision', cls, emo_baseline['precision'][i], emo_enhanced['precision'][i]])
        emo_rows.append(['Recall', cls, emo_baseline['recall'][i], emo_enhanced['recall'][i]])
        emo_rows.append(['F1 Score', cls, emo_baseline['f1'][i], emo_enhanced['f1'][i]])
        
    draw_table("Emotion Dataset Performance Comparison", emo_rows, columns, "emotion_table.png")

def draw_summary_table(title, data_rows, columns, filename):
    fig, ax = plt.subplots(figsize=(8, len(data_rows) * 0.6 + 1.5))
    ax.axis('off')
    
    table_data = []
    cell_colors = []
    
    for row in data_rows:
        metric, base, enh = row
        base_val, enh_val = float(base), float(enh)
        
        base_str = f"{base_val:.4f}" if "Accuracy" in metric else f"{base_val:.2f}"
        enh_str = f"{enh_val:.4f}" if "Accuracy" in metric else f"{enh_val:.2f}"
        
        # Highlight higher score
        row_colors = ['white', 'white', 'white'] 
        if base_val > enh_val:
            row_colors[1] = '#C6F6D5' # Green highlight
            row_colors[2] = '#FED7D7' # Red lowlight
        elif enh_val > base_val:
            row_colors[1] = '#FED7D7'
            row_colors[2] = '#C6F6D5'
            
        table_data.append([metric, base_str, enh_str])
        cell_colors.append(row_colors)
        
    table = ax.table(cellText=table_data, colLabels=columns, cellColours=cell_colors, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)
    
    # Style Header
    for j, col in enumerate(columns):
        cell = table[0, j]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2D3748')
        
    # Make all metric rows bold
    for i in range(len(data_rows)):
        for j in range(3):
            table[i+1, j].set_text_props(weight='bold')
                
    plt.title(title, fontsize=18, fontweight='bold', pad=20, color='#2D3748')
    plt.tight_layout()
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_tables():
    columns = ['Metric', 'Baseline TF-IDF', 'Enhanced Pipeline']
    
    # SST-2 Summary
    sst2_rows = [
        ['Accuracy', sst2_baseline['accuracy'], sst2_enhanced['accuracy']],
        ['Macro F1', sst2_baseline['macro_f1'], sst2_enhanced['macro_f1']],
    ]
    draw_summary_table("SST-2 Overall Performance", sst2_rows, columns, "sst2_summary_table.png")
    
    # Emotion Summary
    emo_rows = [
        ['Accuracy', emo_baseline['accuracy'], emo_enhanced['accuracy']],
        ['Macro F1', emo_baseline['macro_f1'], emo_enhanced['macro_f1']],
    ]
    draw_summary_table("Emotion Dataset Overall Performance", emo_rows, columns, "emotion_summary_table.png")

if __name__ == "__main__":
    print("Generating presentation plots...")
    plot_single_dataset_metrics("SST-2 Dataset", sst2_baseline['accuracy'], sst2_enhanced['accuracy'], 
                                sst2_baseline['macro_f1'], sst2_enhanced['macro_f1'], "sst2_bar_metrics.png")
    plot_single_dataset_metrics("Emotion Dataset", emo_baseline['accuracy'], emo_enhanced['accuracy'], 
                                emo_baseline['macro_f1'], emo_enhanced['macro_f1'], "emotion_bar_metrics.png")
    plot_both_datasets_metrics()
    
    generate_tables()
    generate_summary_tables()
    print("✅ Plots and Tables saved to the 'plots' directory.")
