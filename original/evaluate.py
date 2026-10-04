import re
import json
import os
import argparse


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
    if "DCFA" in eval_file:
        pattern = r"Prediction for ([^:]+\.json):(.*?)(?=Prediction for |\Z)"
    else:
        pattern = r"---\s*Analyzing File:\s*([^\n]+?)\s*---\s*(.*?)(?=(?:---\s*Analyzing File:)|\Z)"
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
            return str(mistake_agent), str(mistake_step)
        else:
            print(f"Warning: 'mistake_agent' or 'mistake_step' key missing in {labeled_json}")
            return None, None
    except FileNotFoundError:
        print(f"Error: Actual data file not found during read: {labeled_json}")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {labeled_json}")
        return None, None
    except Exception as e:
        print(f"Error reading actual data from {labeled_json}: {e}")
        return None, None

def evaluate_accuracy(predictions, data_path, total_files, alpha=0.1):
    correct_agent = 0
    correct_step = 0
    correct_agent_step = 0
    correct_agent_wrong_step = 0
    correct_step_wrong_agent = 0

    not_first_attribution = 0
    not_first_attribution_dis = []
    files_evaluated = 0

    # Hit@k metrics
    hit_at_1 = 0
    hit_at_2 = 0
    hit_at_alphaL = 0
    lengths = []

    if total_files == 0:
        print("Error: No JSON files found in the data path to evaluate against.")
        return 0.0, 0.0, 0.0

    print(f"\n--- Starting Evaluation ---")
    print(f"Total reference JSON files found in {data_path}: {total_files}")
    print(f"Predictions available for {len(predictions)} files.")
    print("=======================================")

    for idx, pred in predictions.items():
        labeled_file = os.path.join(data_path, f"{idx}")

        if os.path.exists(labeled_file):
            files_evaluated += 1
            actual_agent, actual_step = read_actual_data(labeled_file)
            actual_agent_flag = False
            actual_step_flag = False
            if actual_agent is not None and actual_step is not None:
                pred_step = int(pred['predicted_step'])
                gt_step = int(actual_step)
                not_first_attribution_dis.append(abs(pred_step - gt_step))
                # ----- Accuracy -----
                if actual_agent == pred['predicted_agent']:
                    actual_agent_flag = True
                    correct_agent += 1
                if gt_step == pred_step:
                    actual_step_flag = True
                    correct_step += 1
                if actual_agent_flag and actual_step_flag:
                    correct_agent_step += 1
                elif not actual_step_flag:        
                    correct_agent_wrong_step += 1
                    if gt_step < pred_step:
                        not_first_attribution += 1
                        
                elif not actual_agent_flag and actual_step_flag:
                    correct_step_wrong_agent += 1

                # ----- Hit@k -----
                try:
                    with open(labeled_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        log_len = data.get("total_steps") or data.get("num_steps") or max(gt_step, pred_step)
                except:
                    log_len = max(gt_step, pred_step)
                lengths.append(log_len)

                # Hit@1
                d1 = 1 # max(1, int(0.1 * log_len))
                if abs(pred_step - gt_step) <= d1:
                    hit_at_1 += 1
                # Hit@2
                d2 = 3 # max(1, int(0.2 * log_len))
                if abs(pred_step - gt_step) <= d2:
                    hit_at_2 += 1
                # Hit@αL
                d3 = 5 #max(1, int(0.3 * log_len))
                if abs(pred_step - gt_step) <= d3:
                    hit_at_alphaL += 1

            else:
                 print(f"Skipping evaluation for {idx} due to issues reading actual data.")

        else:
            print(f"Warning: Labeled file not found for prediction key '{idx}': {labeled_file}")

    print("\n--- Evaluation Summary ---")
    print(f"Total reference files in data_path: {total_files}")
    print(f"Predictions parsed from eval file:  {len(predictions)}")
    print(f"Files evaluated (prediction found & actual data read): {files_evaluated}")
    print(f"Correct Agent Predictions: {correct_agent}")
    print(f"Correct Step Predictions:  {correct_step}")
    print(f"Correct Agent & Step Predictions: {correct_agent_step}")
    print(f"Correct Agent but Wrong Step Predictions: {correct_agent_wrong_step}")
    print(f"Correct Step but Wrong Agent Predictions: {correct_step_wrong_agent}")

    print(f"Not First Attribution Cases: {not_first_attribution}")
    print(f"Average Delay in Not First Attribution: {sum(not_first_attribution_dis)/total_files if total_files > 0 else 0:.2f}")

    agent_accuracy = int(correct_agent ) / total_files * 100 if total_files > 0 else 0.0
    step_accuracy = int(correct_agent_step) / total_files * 100 if total_files > 0 else 0.0
    not_first_attribution_rate = (not_first_attribution / (correct_agent_wrong_step)) * 100 if (correct_agent_wrong_step) > 0 else 0.0

    # ----- Hit@k 输出 -----
    hit1_rate = (hit_at_1 / total_files) * 100 if files_evaluated > 0 else 0.0
    hit2_rate = (hit_at_2 / total_files) * 100 if files_evaluated > 0 else 0.0
    hitalpha_rate = (hit_at_alphaL / total_files) * 100 if files_evaluated > 0 else 0.0

    print("\n--- Hit@k Metrics ---")
    print(f"Hit@1: {hit1_rate:.2f}%")
    print(f"Hit@3: {hit2_rate:.2f}%")
    print(f"Hit@5: {hitalpha_rate:.2f}%")

    return agent_accuracy, step_accuracy, not_first_attribution_rate

def main(args):    
    data_path = args.data_path
    eval_file = args.eval_file

    if not os.path.isdir(data_path):
        print(f"Error: Data directory not found at {data_path}")
        actual_total_files = 0
    else:
        try:
            json_files_in_data_path = [
                f for f in os.listdir(data_path)
                if f.endswith('.json') and os.path.isfile(os.path.join(data_path, f))
            ]
            actual_total_files = len(json_files_in_data_path)
        except Exception as e:
            print(f"Error reading data directory {data_path}: {e}")
            actual_total_files = 0

    predictions = read_predictions(eval_file)
    if args.max_file > 0:
        predictions = {
            k: v for k, v in predictions.items()
            if int(''.join(filter(str.isdigit, k))) < args.max_file
        }

    agent_accuracy, step_accuracy, not_first_attribution_rate = evaluate_accuracy(predictions, data_path, actual_total_files)

    print("\n--- Final Accuracy Results ---")
    print(f"Evaluation File: {eval_file}")
    print(f"Data Path:       {data_path}")
    print(f"Agent Accuracy: {agent_accuracy:.2f}%")
    print(f"Step Accuracy:  {step_accuracy:.2f}%")
    print(f"Not First Attribution rate: {not_first_attribution_rate:.2f}%")
    print(f"(Accuracy calculated based on {actual_total_files} total files in data path)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agent and step prediction accuracy from an evaluation log file.")
    parser.add_argument(
        "--eval_file",
        type=str,
        # required=True,
        default='outputs/<algorithm>_<llm id>_<dataset>.txt',
        help="alg_generated or handcrafted."
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