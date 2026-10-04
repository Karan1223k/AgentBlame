import re
import json
import os
import argparse
import matplotlib.pyplot as plt


DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Who&When')

def read_predictions(eval_file):
    if not os.path.exists(eval_file):
        print(f"Error: Evaluation file not found at {eval_file}")
        return {}

    try:
        with open(eval_file, 'r', encoding='utf-8') as file:
            data = file.read()
    except Exception as e:
        print(f"Error reading evaluation file {eval_file}: {e}")
        return {}

    predictions = {}
    # if "CMA" in eval_file:
    pattern = r"Prediction for ([^:]+\.json):(.*?)(?=Prediction for |\Z)"
    # else:
    #     pattern = r"---\s*Analyzing File:\s*([^\n]+?)\s*---\s*(.*?)(?=(?:---\s*Analyzing File:)|\Z)"
    blocks = re.finditer(pattern, data, re.DOTALL)
    parsed_count = 0

    for block in blocks:
        content = block.group(2).strip()
        idx = block.group(1).strip()
        agent_name_match = re.search(r"Agent Name:\s*([\w_]+)", content, re.IGNORECASE)
        step_number_match = re.search(r"Step Number:\s*(\d+)", content, re.IGNORECASE)

        if step_number_match:
            agent_name = agent_name_match.group(1) if agent_name_match else "Unknown_Agent"
            step_number = step_number_match.group(1)
            predictions[idx] = {
                'predicted_agent': agent_name,
                'predicted_step': f"{step_number}"
            }
            parsed_count += 1
        else:
            print(f"Warning: Could not parse Agent Name/Step Number for {idx} in {eval_file}")
            
    print(f"--- Predictions Read from {eval_file} ---")
    print(f"Successfully parsed predictions for {parsed_count} files.")
    print("=======================================")
    return predictions


def read_actual_data(labeled_json):
    try:
        with open(labeled_json, 'r', encoding='utf-8') as file:
            data = json.load(file)
        mistake_agent = data.get('mistake_agent')
        mistake_step = data.get('mistake_step')
        if mistake_agent is not None and mistake_step is not None:
            return str(mistake_agent), str(mistake_step), data
        else:
            print(f"Warning: 'mistake_agent' or 'mistake_step' key missing in {labeled_json}")
            return None, None, None
    except Exception as e:
        print(f"Error reading actual data from {labeled_json}: {e}")
        return None, None, None


def evaluate_accuracy(predictions, data_path, total_files):
    level_ranges = {
        "Level 1 (5–17)": (5, 17),
        "Level 2 (19–29)": (19, 29),
        "Level 3 (31–49)": (31, 49),
        "Level 4 (51–91)": (51, 91),
        "Level 5 (93–130)": (93, 130)
    }

    level_stats = {level: {"correct": 0, "total": 0} for level in level_ranges}

    files_evaluated = 0
    for idx, pred in predictions.items():
        labeled_file = os.path.join(data_path, f"{idx}")
        if not os.path.exists(labeled_file):
            continue

        actual_agent, actual_step, data = read_actual_data(labeled_file)
        if actual_agent is None or actual_step is None:
            continue

        try:
            log_len = len(data.get("history"))
        except:
            log_len = int(actual_step)

        pred_step = int(pred["predicted_step"])
        gt_step = int(actual_step)

        #
        level_name = None
        for name, (low, high) in level_ranges.items():
            if low <= log_len <= high:
                level_name = name
                break
        if not level_name:
            continue  #

        files_evaluated += 1
        level_stats[level_name]["total"] += 1
        if abs(pred_step - gt_step) == 0:
            level_stats[level_name]["correct"] += 1

    # 
    level_accuracies = {}
    for name, stats in level_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        acc = (correct / total * 100) if total > 0 else 0.0
        level_accuracies[name] = acc

    print("\n--- Accuracy by Log Length Level ---")
    for level, acc in level_accuracies.items():
        total = level_stats[level]["total"]
        print(f"{level}: {acc:.2f}% (based on {total} logs)")

    #
    plt.figure(figsize=(8, 5))
    levels = list(level_accuracies.keys())
    accs = [level_accuracies[l] for l in levels]
    plt.plot(levels, accs, marker='o', linestyle='-', linewidth=2)
    plt.title("Step Accuracy across Log Length Levels")
    plt.xlabel("Log Length Level")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    return level_accuracies


def main(args):
    data_path = args.data_path
    eval_file = args.eval_file

    if not os.path.isdir(data_path):
        print(f"Error: Data directory not found at {data_path}")
        return

    try:
        json_files_in_data_path = [
            f for f in os.listdir(data_path)
            if f.endswith('.json') and os.path.isfile(os.path.join(data_path, f))
        ]
        total_files = len(json_files_in_data_path)
    except Exception as e:
        print(f"Error reading data directory {data_path}: {e}")
        total_files = 0

    predictions = read_predictions(eval_file)
    if args.max_file > 0:
        predictions = {
            k: v for k, v in predictions.items()
            if int(''.join(filter(str.isdigit, k))) < args.max_file
        }

    level_accuracies = evaluate_accuracy(predictions, data_path, total_files)
    print("\n--- Final Results ---")
    for level, acc in level_accuracies.items():
        print(f"{level}: {acc:.2f}%")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate step accuracy across different log length levels.")
    parser.add_argument(
        "--eval_file",
        type=str,
        default='outputs/DCFA_<llm name>_<dataset>.txt',
        help="Evaluation file."
    )
    args = parser.parse_args()
    if 'alg' in os.path.basename(args.eval_file).lower():
        args.data_path = os.path.join(DATA_ROOT, 'Algorithm-Generated')
    elif 'hand' in os.path.basename(args.eval_file).lower():
        args.data_path = os.path.join(DATA_ROOT, 'Hand-Crafted')
    else:
        args.data_path = os.path.join(DATA_ROOT, 'Algorithm-Generated')
    args.max_file = 0
    main(args)