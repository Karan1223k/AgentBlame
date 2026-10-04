import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time
from datetime import datetime
import re
import torch.nn.functional as F
import logging
from Lib.device import get_default_device, get_dtype
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# =========================
# Configuration
# =========================
load_dotenv()
DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_K = 3  # For inter-agent causality
DEFAULT_MODEL = "Qwen3-8B"  # Default LLM model
EMBED_MODEL = "text-embedding-3-large"

KNOWN_GPT_MODELS = {"gemini-2.5-pro", "DMXAPI-HuoShan-DeepSeek-R1-671B-64k", "gpt-5", "qwen3-235b-a22b"}
LOCAL_MODEL_ALIASES = {"qwen3-coder-30b-a3b-instruct", "deepseek-r1-32b"}
ALL_MODELS = list(KNOWN_GPT_MODELS | LOCAL_MODEL_ALIASES)

LOCAL_MODEL_MAP = {
    "qwen3-coder-30b-a3b-instruct": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "deepseek-r1-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
}

round_float = 3

# =========================
# LLM & Embedding Clients
# =========================
def get_llm_client(args):
    if args.model in KNOWN_GPT_MODELS:
        logger.info(f"Initializing OpenAI client for model: {args.model}")
        api_key = args.api_key
        base_url = args.base_url
        if not api_key:
            raise ValueError("OpenAI API key is required. Use --api_key or set OPENAI_API_KEY.")
        return OpenAI(api_key=api_key, base_url=base_url), "gpt"

def get_vllm_client(args):
    logger.info(f"Initializing OpenAI client for model: {args.LCCE}")
    api_key = args.vllm_api_key
    base_url = args.vllm_base_url
    if not api_key:
        raise ValueError("OpenAI API key is required. Use --api_key or set OPENAI_API_KEY.")
    return OpenAI(api_key=api_key, base_url=base_url), "vllm"
    
def get_local_client(args):
    model_id = os.path.join(args.local_model_path, LOCAL_MODEL_MAP[args.LCCE])
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Multi-GPU sharding only makes sense on CUDA; on Mac (MPS) / CPU load onto a single device
    device_map = "auto" if args.device.startswith("cuda") else {"": args.device}
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=get_dtype(args.device),
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return pipe, "local"

def get_embedding(text, client, model=DEFAULT_EMBED_MODEL):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def get_prompt(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def extract_fields_from_broken_json(text: str, expected_keys=None):
    result = {}
    if not expected_keys:
        return result

    for key in expected_keys:
        pattern = re.compile(rf'"{key}"\s*:\s*"([^"]*?)"', re.DOTALL)
        match = pattern.search(text)
        if match:
            result[key] = match.group(1)
    return result


def safe_llm_call(client, model_type, prompt, max_retries=2,
                  expected_keys=None, is_json=True, temperature=0):
    last_exception = None

    for attempt in range(max_retries):
        try:
            # ==================== 调用模型 ====================
            if model_type == "gpt":
                response_format = {"type": "json_object"} if is_json else {"type": "text"}
                response = client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    response_format=response_format
                )
                result_text = response.choices[0].message.content
            elif model_type == "vllm":
                response_format = {"type": "json_object"} if is_json else {"type": "text"}
                response = client.chat.completions.create(
                    model=DEFAULT_VLLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    response_format=response_format
                )
                result_text = response.choices[0].message.content
            else:  # 本地模型
                if is_json:
                    prompt = (
                        prompt
                        + "\n\nOutput results strictly in JSON format and do not include additional text. For example:\n"
                        + "{\n  \"key1\": \"value1\",\n  \"key2\": \"value2\"\n}"
                    )
                with torch.inference_mode():
                    response = client(
                        prompt,
                        max_new_tokens=4096*2 ,
                        do_sample=(temperature > 0),
                        temperature=temperature,
                        pad_token_id=client.tokenizer.eos_token_id,
                        return_full_text=False,
                    )
                result_text = response[0]["generated_text"]

                if is_json:
                    json_start = result_text.find('{')
                    json_end = result_text.rfind('}') + 1
                    if json_start != -1 and json_end != 0:
                        result_text = result_text[json_start:json_end]

            # ==================== 纯文本模式 ====================
            if not is_json:
                if not result_text.strip():
                    raise ValueError("LLM returned an empty response.")
                return result_text

            # ==================== JSON 模式 ====================
            try:
                parsed_json = json.loads(result_text)
            except json.JSONDecodeError:
                # 
                logger.warning(f"JSON parse failed at attempt {attempt+1}, extracting fields manually.")
                parsed_json = extract_fields_from_broken_json(result_text, expected_keys)
                if not parsed_json:
                    raise ValueError("JSON parse failed and no expected keys found.")

            # ==================== expected_keys ====================
            if expected_keys:
                missing_keys = [k for k in expected_keys if k not in parsed_json]
                if missing_keys:
                    logger.warning(f"Missing keys in LLM response: {missing_keys}")
            return parsed_json

        except (ValueError, KeyError) as e:
            last_exception = e
            logger.warning(f"LLM call attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt + 1 < max_retries:
                time.sleep(2)
            else:
                logger.error("Max retries reached. Raising error.")

    raise last_exception

# =========================
# Module 1: Event Extraction
# =========================
def convert_log_to_json(log_content):
    output = {}

    for idx, entry in enumerate(log_content):
        agent_name = entry.get("name", "Unknown")
        if agent_name == "Unknown":
            agent_name = entry.get("role", "Unknown")
        content = entry.get("content", "")

        # 
        event_id = f"event_{idx}_1"

        output[str(idx)] = {
            "agent_name": agent_name.split()[0],
            "event_dict": {
                event_id: content
            }
        }

    return output

def extract_events(log_content, output_path, client, model_type="gpt", max_retries=3):
    logger.info("--- Running: Event Extraction ---")
    log_content = convert_log_to_json(log_content)
    prompt_template = get_prompt(os.path.join(current_dir, "agent_prompt/event_extraction.txt"))
    prompt = prompt_template.format(
        event_log=json.dumps(log_content, indent=2, ensure_ascii=False)
    )
    expected_keys = ["extracted_events", "mistake_event_reasons"]
    mistake_event_reasons = safe_llm_call(client, model_type, prompt, max_retries=max_retries, is_json=True)
    
    events = {
        "extracted_events": log_content,
        "mistake_event_reasons": mistake_event_reasons}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    logger.info(f"Extracted events saved to {output_path}")
    return events
# =========================
# Module 2: Causal Graph Discovery on mistake events
# =========================
def discover_intra_agent_causality(events, output_path, client, model_type="gpt", max_retries=3):
    logger.info("--- Running: Intra-Agent Causality Discovery ---")
    extract_events_json = events["extracted_events"]
    mistake_reason = events["mistake_event_reasons"]

    prompt_template = get_prompt(os.path.join(current_dir, "agent_prompt/event_causality.txt"))
    prompt = prompt_template.format(ALL_EVENT_JSON=extract_events_json, MISTAKE_EVENTS_REASONS_JSON=mistake_reason)  # K=0 for intra-agent
    all_events_causality = safe_llm_call(client, model_type, prompt, max_retries=max_retries, is_json=True)
    mistake_events_causality = {}
    if "mistake_event_reasons" in events:
        for event_id, reason in events["mistake_event_reasons"].items():
            mistake_event_step = event_id.split('_')[1]  # 从 event_id 中提取 step_id
            mistake_events_causality[f"{mistake_event_step}"] = {
                        "event_dict": {
                            event_id: {
                                "cause": all_events_causality.get(mistake_event_step, {}).get("event_dict", {}).get(event_id, {}).get("cause", []),
                                "effect": all_events_causality.get(mistake_event_step, {}).get("event_dict", {}).get(event_id, {}).get("effect", [])
                            }
                        }
                    }
    causality = {}
    causality['all_events_causality'] = all_events_causality
    causality["mistake_events_causality"] = mistake_events_causality

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(causality, f, ensure_ascii=False, indent=2)
    logger.info(f"Intra-agent causality saved to {output_path}")
    return causality

def discover_inter_agent_causality(events_path, output_path, client, k, model_type="gpt", max_retries=3):
    logger.info("--- Running: Inter-Agent Causality Discovery ---")
    with open(events_path, "r", encoding="utf-8") as f:
        mistake_events_json = f.read()

    prompt_template = get_prompt(os.path.join(current_dir, "agent_prompt/mistake_event_causality.txt"))
    prompt = prompt_template.format(MISTAKE_EVENT_JSON=mistake_events_json, K=k)

    causality = safe_llm_call(client, model_type, prompt, max_retries=max_retries, is_json=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(causality, f, ensure_ascii=False, indent=2)
    logger.info(f"Inter-agent causality saved to {output_path}")
    return causality

def merge_causal_graphs(intra_path, inter_path, output_path):
    logger.info("--- Merging Causal Graphs ---")
    with open(intra_path, 'r') as f:
        intra_data = json.load(f)
    with open(inter_path, 'r') as f:
        inter_data = json.load(f)

    # A simple merge, assuming the structure is the same and we can just update
    # A more robust implementation would merge the cause/effect lists
    merged_data = intra_data.copy()
    for step_id, step_content in inter_data.items():
        if step_id in merged_data:
            for event_id, event_content in step_content["event_dict"].items():
                if event_id in merged_data[step_id]["event_dict"]:
                    # Merge cause/effect lists, avoiding duplicates
                    for key in ["cause", "effect"]:
                        existing = set(merged_data[step_id]["event_dict"][event_id].get(key, []))
                        new = set(event_content.get(key, []))
                        merged_data[step_id]["event_dict"][event_id][key] = list(existing.union(new))
                else:
                    merged_data[step_id]["event_dict"][event_id] = event_content
        else:
            merged_data[step_id] = step_content

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Merged causal graph saved to {output_path}")
    return merged_data

# =========================
# Module 3: Causal Mediation Analysis
# =========================
def identify_initial_mistake(extracted_event, output_path, question, causal_graph, client, ground_truth=None, model_type="gpt", max_retries=3):
    logger.info("--- Running: Initial Mistake Identification ---")
    mistake_events_reasons = extracted_event['mistake_event_reasons']
    all_events = extracted_event['extracted_events']
    prompt_template = get_prompt(os.path.join(current_dir, "agent_prompt/failure_attribution.txt"))
    prompt = prompt_template.format(QUESTION=question, GROUND_TRUTH=ground_truth, ALL_EVENTS_JSON=json.dumps(all_events, indent=2), MISTAKE_EVENTS_REASONS_JSON=json.dumps(mistake_events_reasons, indent=2), CAUSAL_GRAPH_JSON=json.dumps(causal_graph, indent=2))
    expected_keys = ["log_answer", "root_cause_event", "event_step", "reason"]
    mistake_info = safe_llm_call(client, model_type, prompt, max_retries=max_retries, expected_keys=expected_keys, is_json=True)
    
    logger.info(f"Identified initial mistake event: {mistake_info['root_cause_event']} in step {mistake_info['event_step']}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mistake_info, f, ensure_ascii=False, indent=2)
    logger.info(f"Initial root cause event saved to {output_path}")
    return mistake_info

def identify_final_mistake(useful_causal_event_list, extracted_event, output_path, question, causal_graph, client, ground_truth=None, model_type="gpt", max_retries=3):
    logger.info("--- Running: Final Mistake Identification ---")
    # Normalize keys to a list of keys
    useful_causal_event_list = dict(
                                sorted(
                                    useful_causal_event_list.items(),
                                    key=lambda x: int(x[0].split("_")[1])
                                )
                            )
    if isinstance(useful_causal_event_list, dict):
        keys = list(useful_causal_event_list.keys())
    elif isinstance(useful_causal_event_list, (list, set, tuple)):
        keys = list(useful_causal_event_list)
    else:
        keys = [useful_causal_event_list]
    logger.info(" -> ".join(keys))
    steps = [key.split("_")[1] for key in keys]
    mistake_events_reasons = extracted_event['mistake_event_reasons']
    all_events = extracted_event['extracted_events']
    useful_cause_events = {}
    for i in all_events.keys():
        if i in steps:            
            event_id = f"event_{i}_1"
            useful_cause_events.update(all_events.get(i, {}))
            if event_id in mistake_events_reasons.keys():
                useful_causal_event_list[event_id] =  useful_causal_event_list.get(event_id, "") +"\n" + mistake_events_reasons[event_id]
    
    prompt_template = get_prompt(os.path.join(current_dir, "agent_prompt/final_failure_attribution.txt"))
    prompt = prompt_template.format(QUESTION=question, 
                                    GROUND_TRUTH=ground_truth, 
                                    CAUSAL_EVENTS_JSON=json.dumps(useful_cause_events, indent=2), 
                                    MISTAKE_EVENTS_REASONS_JSON=json.dumps(useful_causal_event_list, indent=2))
    
    expected_keys = ["root_cause_event", "reason"]
    mistake_info = safe_llm_call(client, model_type, prompt, max_retries=max_retries, expected_keys=expected_keys, is_json=True)
    root_event_step = mistake_info['root_cause_event'].split("_")[1]
    logger.info(f"Identified final mistake event: {mistake_info['root_cause_event']} in step {root_event_step}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mistake_info, f, ensure_ascii=False, indent=2)
    logger.info(f"Final root cause event saved to {output_path}")
    return mistake_info

def correct_event_and_log(flag, events, question, causal_graph, current_root_cause_event, current_root_reason, client, ground_truth=None, model_type="gpt", max_retries=3):
    logger.info(f"--- Correcting event {current_root_cause_event} and subsequent log ---")
    prompt_template = get_prompt(os.path.join(current_dir, "agent_prompt/root_cause_event_correction.txt")) if flag != "Hand-Crafted" else get_prompt(os.path.join(current_dir, "agent_prompt/root_cause_event_correction_hand-crafted.txt"))
    extracted_event = events['extracted_events']
    mistake_reason = events["mistake_event_reasons"]
    prompt = prompt_template.format(
        QUESTION=question,
        GROUND_TRUTH=ground_truth,
        MISTAKE_EVENT_REASONS=json.dumps(mistake_reason, indent=2),
        ALL_EVENTS=json.dumps(extracted_event, indent=2),
        CAUSAL_GRAPH_JSON=json.dumps(causal_graph, indent=2)
    )
    corrected_data = safe_llm_call(client, model_type, prompt, max_retries=max_retries, is_json=True)

    return corrected_data

def run_simulation(modified_log, question, current_root_cause_event, client, model_type="gpt", max_retries=3):
    logger.info("--- Running simulation to get outcome ---")
    prompt_template = get_prompt(os.path.join(current_dir, "agent_prompt/simulation_outcome.txt"))
    prompt = prompt_template.format(QUESTION=question, MODIFIED_LOG=json.dumps(modified_log, indent=2), ROOT_CAUSE_EVENT=json.dumps(current_root_cause_event, indent=2))
    
    outcome = safe_llm_call(client, model_type, prompt, max_retries=max_retries, is_json=True)
    return outcome

def mse(embedding1, embedding2):
    return np.mean((np.array(embedding1) - np.array(embedding2))**2)

def calculate_total_effect(question, log_answer, causal_graph, extracted_events, current_root_cause_event, modified_all_event, ground_truth, client, model_type, args):
    max_retries=args.max_retries
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    embedding_client = OpenAI(api_key=api_key, base_url=base_url)

    # Helper to convert event list back to dict for simulation
    def list_to_event_dict(event_list):
        event_dict = {}
        for event in event_list:
            step_id = event["step_id"]
            event_id = event["event_id"]
            if step_id not in event_dict:
                event_dict[step_id] = {"event_dict": {}}
            
            event_content = {"content": event["content"]}
            if "cause" in event: event_content["cause"] = event["cause"]
            if "effect" in event: event_content["effect"] = event["effect"]
            event_dict[step_id]["event_dict"][event_id] = event_content
        return event_dict

    def loss_align_projection(de_list, log_list, gt_list, device="cpu", alpha=0.5):
        # tensor
        de = torch.tensor(de_list, dtype=torch.float32, device=device)
        log = torch.tensor(log_list, dtype=torch.float32, device=device)
        gt = torch.tensor(gt_list, dtype=torch.float32, device=device)
        if de.dim() == 1:
            de = de.unsqueeze(0)
            log = log.unsqueeze(0)
            gt = gt.unsqueeze(0)
        L_gt = F.cosine_similarity(de, gt, dim=-1)
        v = de - log
        u = gt - log
        L_dir = F.cosine_similarity(v, u, dim=-1)
        L = L_gt
        return L.detach().cpu().item()

    #
    def build_interaction_log(events_dict, agent_name_for_each_step):
        interaction_log = []
        sorted_steps = sorted(events_dict.keys(), key=lambda x: int(x))  # 按 step_id 排序

        for step_key in sorted_steps:
            step_data = events_dict[step_key]
            agent_name = agent_name_for_each_step[int(step_key)]

            # 
            sorted_events = sorted(
                step_data["event_dict"].items(),
                key=lambda x: int(x[0].split('_')[2])
            )

            # 拼接内容
            contents = []
            for event_id, event_data in sorted_events:
                if isinstance(event_data, dict) and "content" in event_data:
                    contents.append(event_data["content"])
                elif isinstance(event_data, str):
                    contents.append(event_data)
                else:
                    contents.append(str(event_data))

            interaction_log.append({
                "step_id": step_key,
                "agent": agent_name,
                "content": " ".join(contents)
            })

        return interaction_log
    
    # Flatten original and modified events for easier manipulation
    agent_name_for_each_step = []
    original_events_list = []
    sorted_steps = sorted(extracted_events.keys(), key=lambda x: int(x))
    for step_key in sorted_steps:
        agent_name_for_each_step.append(extracted_events[step_key].get("agent_name", f"Agent_{step_key}"))
        sorted_events = sorted(extracted_events[step_key]["event_dict"].items(), key=lambda x: int(x[0].split('_')[2]))
        for event_id, event_data in sorted_events:
            original_events_list.append({"event_id": event_id, "step_id": step_key, 'content': event_data})


    modified_events_list = []
    if modified_all_event:
        valid_keys = [k for k in modified_all_event.keys()]
        sorted_steps_mod = sorted(valid_keys, key=lambda x: int(x))
        for step_key in sorted_steps_mod:
            if step_key in modified_all_event and "event_dict" in modified_all_event[step_key]:
                valid_event_keys = [ek for ek in modified_all_event[step_key]["event_dict"].keys() if ek.startswith("event_")]
                sorted_events_mod = sorted(valid_event_keys, key=lambda x: int(x.split('_')[2]))
                for event_id in sorted_events_mod:
                    modified_events_list.append({"event_id": event_id, "step_id": step_key, "content": modified_all_event[step_key]["event_dict"][event_id]})

    mistake_index = int(current_root_cause_event.split('_')[1])
    
    if mistake_index == -1:
        # Fallback for safety, though it should be found.
        logger.info(f"Warning: Mistake event {current_root_cause_event} not found in extracted events list. Effects may be inaccurate.")
        return 0, 0, 0

    ## Total Effect Calculation
    original_events_list = json.loads(json.dumps(original_events_list))  
    corrected_event_content = None
    for event in modified_events_list:
        if event['event_id'] == current_root_cause_event:
            corrected_event_content = event['content']
            break
    if corrected_event_content is None:
        logger.info(f"Warning: Corrected content for {current_root_cause_event} not found. Direct effect will be 0.")
        delta_total_effect = 0
    else:
        # total_effect_log_list = modified_events_list
        # total_effect_log_list[mistake_index] = original_events_list[mistake_index]
        
        # P(Y| do(i))
        total_effect_log_list = modified_events_list[:mistake_index+1] + original_events_list[mistake_index+1:]
        
        directly_related_events = causal_graph.get(str(mistake_index), {}).get("event_dict", {}).get(current_root_cause_event, {}).get("effect", [])
        current_root_cause_event = {current_root_cause_event: directly_related_events}
        
        te_log_dict = list_to_event_dict(total_effect_log_list)
        te_log_dict = build_interaction_log(te_log_dict, agent_name_for_each_step)
        # prompt_emphasis = f"Carefully focus on the correction made to event {current_root_cause_event} and thoroughly analyze its downstream effects along the event chain. Ensure that after this correction, all subsequent events are properly updated to reflect the change. Rigorously examine whether these updates alter the final answer to the QUESTION."
        outcome = run_simulation(te_log_dict, question, current_root_cause_event, client, model_type, max_retries)
        te_outcome = str(outcome['answer'])
        te_outcome_embedding = get_embedding(te_outcome, embedding_client, model=EMBED_MODEL)
        
        ground_truth_embedding = get_embedding(ground_truth, embedding_client, model=EMBED_MODEL)
        log_answer_embedding = get_embedding(log_answer, embedding_client, model=EMBED_MODEL)
        
        total_effect = loss_align_projection(te_outcome_embedding, log_answer_embedding, ground_truth_embedding)
        pre_total_effect_log_list = modified_events_list[:mistake_index] + original_events_list[mistake_index:]
        pre_te_log_dict = list_to_event_dict(pre_total_effect_log_list)
        pre_te_log_dict = build_interaction_log(pre_te_log_dict, agent_name_for_each_step)
        # 
        pre_outcome = run_simulation(pre_te_log_dict, question, None, client, model_type, max_retries)
        pre_te_outcome = str(pre_outcome['answer'])
        # 
        pre_te_outcome_embedding = get_embedding(pre_te_outcome, embedding_client, model=EMBED_MODEL)
        # 
        pre_total_effect = loss_align_projection(pre_te_outcome_embedding, log_answer_embedding, ground_truth_embedding)
        
        delta_total_effect = total_effect - pre_total_effect

    logger.info(f"[DEBUG] Total Effect (after correction) = {total_effect:.4f}")
    logger.info(f"[DEBUG] Total Effect (before correction) = {pre_total_effect:.4f}")
    logger.info(f"[DEBUG] Δ Total Effect = {delta_total_effect:.4f}")

    return delta_total_effect, outcome['reason']

def get_event_candidates(current_event, history_candidates, causal_graph, trace_direction, all_events):
    candidates = set()
    current_step = int(current_event.split('_')[1])
    #
    cause_events = causal_graph.get(str(current_step), {}).get('event_dict', {}).get(current_event, {}).get('cause', [])
    effect_events = causal_graph.get(str(current_step), {}).get('event_dict', {}).get(current_event, {}).get('effect', [])
    if trace_direction == "forward":  # Trace forward
        for event_id in cause_events:
            if event_id not in history_candidates:
                candidates.add(event_id)

    elif trace_direction == "backward":  # Trace backward
        for event_id in effect_events:
            if event_id not in history_candidates:
                candidates.add(event_id)
    else:  #
        for event_id in cause_events:
            if event_id not in history_candidates:
                candidates.add(event_id)
        for event_id in effect_events:
            if event_id not in history_candidates:
                candidates.add(event_id)
    
    return list(candidates)


def search_on_direction(args, current_root_cause_event, trace_direction, history_candidates, causal_graph, events, modified_all_event, question, log_answer, output_dir, file_basename, ground_truth, llm_client, model_type):
    useful_causal_event_list = dict()
    pre_event_step = current_root_cause_event.split("_")[1]
    for i in range(args.max_iterations): 
        max_total_effect = -float('inf')

        logger.info(f"\n--- Iteration {i+1}: Analyzing {trace_direction} events of '{current_root_cause_event} ---")
        event_candidates = get_event_candidates(current_root_cause_event, history_candidates, causal_graph['all_events_causality'], trace_direction, events["extracted_events"])
        
        best_candidate = None
        should_break = False
        

        #
        if not event_candidates:
            logger.info(f"No more candidates in the cause of {current_root_cause_event} to explore. Stopping forward iteration.")
            should_break = True
        else:
            for candidate in event_candidates:
                if candidate in history_candidates:
                    continue
                history_candidates.add(candidate)
                logger.info(f"  - Evaluating candidate: {candidate}")
                try:
                    candidate_step = candidate.split('_')[1]  # Assuming event_id format is "event_<step>_<id>"
                    mod_cand_event = modified_all_event
                    candidate_total_effect, mistake_reason = calculate_total_effect(
                        question, log_answer, causal_graph['all_events_causality'], events["extracted_events"], candidate, mod_cand_event, ground_truth, llm_client, model_type, args
                    )
                except Exception as e:
                    logger.info(f"Warning: Could not evaluate candidate '{candidate}': {e}. Skipping.")
                    continue

                if round(candidate_total_effect, round_float) > 0 and ((round(candidate_total_effect, round_float) > round(max_total_effect, round_float)) or \
                    (trace_direction == "forward" and round(candidate_total_effect, round_float) == round(max_total_effect, round_float) and int(candidate_step) < int(pre_event_step)) or \
                    (trace_direction == "backward" and round(candidate_total_effect, round_float) == round(max_total_effect, round_float) and int(candidate_step) > int(pre_event_step))):
                    max_total_effect = candidate_total_effect
                    pre_event_step = candidate_step
                    best_candidate = (candidate, candidate_step, mistake_reason)

            
            if best_candidate:
                logger.info(f"Found better candidate '{best_candidate[0]}' with effect {max_total_effect:.4f} (vs {max_total_effect:.4f}). Updating current event.")
                current_root_cause_event = best_candidate[0]
                #
                useful_causal_event_list[current_root_cause_event] = best_candidate[2]

                effects_output_path = os.path.join(output_dir, f"{file_basename}/{trace_direction}_effects_iter_{i+1}.json")
                os.makedirs(os.path.dirname(effects_output_path), exist_ok=True)
                with open(effects_output_path, "w", encoding="utf-8") as f:
                    json.dump({"event_id": current_root_cause_event, "total_effect": max_total_effect}, f, ensure_ascii=False, indent=2)
            else:
                logger.info("No candidate has a higher total effect. Finalizing root cause.")
                should_break = True

        if should_break:
            break
    
    return useful_causal_event_list, history_candidates

# =========================
# Main Orchestrator
# =========================
def main(args):
    llm_client, model_type = get_llm_client(args)
    if args.local_llm_type == "local":
        lcce_client, lcce_type = get_local_client(args)
    else:
        lcce_client, lcce_type = get_vllm_client(args)
    global DEFAULT_MODEL, DEFAULT_VLLM_MODEL
    DEFAULT_MODEL = args.model
    DEFAULT_VLLM_MODEL = args.LCCE

    # If the output is the default json, change it to a txt file based on params
    input_dir_name = os.path.basename(os.path.normpath(args.input_dir))
    if 'attribution_result.json' in args.output:
        args.output = os.path.join(current_dir, f"outputs/DCFA_{args.model}_{input_dir_name}.txt")

    # Ensure output directory exists and prepare the output file
    output_dir = os.path.join(current_dir, f"DCFA_outputs/{input_dir_name}/{args.model}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all json files in the input directory
    if os.path.isdir(args.input_dir):
        log_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.json')]
        def extract_num(path):
            filename = os.path.basename(path) 
            match = re.search(r'(\d+)\.json$', filename)
            return int(match.group(1)) if match else -1 

        log_files = sorted(log_files, key=extract_num)
    else:
        logger.info(f"Error: Input directory not found at {args.input_dir}")
        return
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as f:
        f.write("--------------------\n\n")
    
    for log_file in tqdm(log_files, desc="Processing log files"):
        logger.info(f"\n\n--- Starting Analysis for {log_file} ---")
        file_basename = os.path.splitext(os.path.basename(log_file))[0]

        # Define file paths for this specific log
        events_output_path = os.path.join(output_dir, f"{file_basename}/events.json")
        intra_causality_path = os.path.join(output_dir, f"{file_basename}/intra_causality.json")
        mistake_info_path = os.path.join(output_dir, f"{file_basename}/mistake_info.json")
        final_mistake_info_path = os.path.join(output_dir, f"{file_basename}/final_mistake_info.json")
        causal_graph_path = os.path.join(output_dir, f"{file_basename}/causal_graph.json")
        history_str_path = os.path.join(output_dir, f"{file_basename}/root_cause_trace.txt")
                
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from input file {log_file}. Skipping.")
            continue
        question = log_data.get("question", "")
        ground_truth = log_data.get("ground_truth", "")
        log_content = log_data.get("history", [])
        # Defaults used by the error handler if a step fails before a prediction is made
        final_event_step, final_root_reason = 0, "Analysis failed before a root cause was identified."

        try: # Added try-except block for safe_llm_call exceptions
            # 1. Event Extraction
            events = None
            if os.path.exists(events_output_path):
                try:
                    with open(events_output_path, 'r', encoding="utf-8") as f:
                        events = json.load(f)
                    logger.info(f"Loaded cached events from {events_output_path}")
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON from {events_output_path}. Regenerating...")
                    events = None
            
            if events is None:
                events = extract_events(log_content, events_output_path, llm_client, model_type, args.max_retries)
            
            # 2. Causal Graph Discovery for All Events
            intra_causality = None
            if os.path.exists(intra_causality_path):
                try:
                    with open(intra_causality_path, 'r', encoding="utf-8") as f:
                        intra_causality = json.load(f)
                    logger.info(f"Loaded cached intra-agent causality from {intra_causality_path}")
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON from {intra_causality_path}. Regenerating...")
                    intra_causality = None
            
            if intra_causality is None:
                intra_causality = discover_intra_agent_causality(events, intra_causality_path, llm_client, model_type, args.max_retries)
            
            # causal graph for mistake events
            causal_graph = intra_causality
            all_events_causality = causal_graph.get("all_events_causality", {})
            # mistake_events_causality = causal_graph.get("mistake_events_causality", {})
            # 3. Causal Mediation Analysis 
            # 3.1 Identify Initial Mistake based on causal graph of the extracted mistake events
            mistake_info = None
            if os.path.exists(mistake_info_path):
                try:
                    with open(mistake_info_path, 'r', encoding="utf-8") as f:
                        mistake_info = json.load(f)
                    logger.info(f"Loaded cached initial mistake_info from {mistake_info_path}")
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON from {mistake_info_path}. Regenerating...")
                    mistake_info = None
            
            if mistake_info is None:
                mistake_info = identify_initial_mistake(
                    events, mistake_info_path, question, all_events_causality, llm_client, ground_truth, model_type, args.max_retries
                )
            
            current_root_cause_event, current_root_reason, log_answer = mistake_info["root_cause_event"], mistake_info["reason"], mistake_info["log_answer"]
            final_root_reason = current_root_reason
            logger.info(f"\n--- Initial Root Cause for {file_basename}: {current_root_cause_event} ---")
            
            final_event_step = current_root_cause_event.split('_')[1]  # Assuming event_id format is "event_<step>_<id>"
            final_mistake_agent = log_content[int(final_event_step)].get('role', 'Unknown').split(' ')[0] if "Hand-Crafted" in args.input_dir else log_content[int(final_event_step)].get('name', 'Unknown')
            flag = "Hand-Crafted" if "Hand-Crafted" in args.input_dir else "Algorithm-Generated"
            final_mistake_info = None
            if os.path.exists(final_mistake_info_path):
                try:
                    with open(final_mistake_info_path, 'r', encoding="utf-8") as f:
                        final_mistake_info = json.load(f)
                    logger.info(f"Loaded cached initial mistake_info from {final_mistake_info_path}")
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON from {final_mistake_info_path}. Regenerating...")
                    final_mistake_info = None
            
            if final_mistake_info is None:
                correction_output_path = os.path.join(output_dir, f"{file_basename}/correction_iter.json")
                correction_data = None
                if os.path.exists(correction_output_path):
                    try:
                        with open(correction_output_path, 'r', encoding="utf-8") as f:
                            correction_data = json.load(f)
                        logger.info(f"Loaded cached correction from {correction_output_path}")
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode JSON from {correction_output_path}. Regenerating...")
                        correction_data = None

                if correction_data:
                    modified_all_event = correction_data
                else:
                    modified_all_event = correct_event_and_log(
                        flag, events, question, all_events_causality, current_root_cause_event, current_root_reason, lcce_client, ground_truth, lcce_type, args.max_retries
                    )
                    os.makedirs(os.path.dirname(correction_output_path), exist_ok=True)
                    with open(correction_output_path, "w", encoding="utf-8") as f:
                        json.dump(modified_all_event, f, ensure_ascii=False, indent=2)

                # 
                useful_causal_event_list = {current_root_cause_event: "It's the initial root cause event selected by the LLM. You need to check it and find the true root cause event."}
                # useful_causal_event_list = {current_root_cause_event: "The event above represents the LLM’s initial hypothesis for the root cause. It must be validated against logs, the causal graph and ground truth; report the adjudicated root cause with supporting and contradictory evidence. You need to check it and find the true root cause event."}

                #
                history_candidates = set()
                forward_useful_causal_event_list, history_candidates = search_on_direction(
                    args, current_root_cause_event, "forward", history_candidates,
                    causal_graph, events, modified_all_event, question, log_answer,
                    output_dir, file_basename, ground_truth, lcce_client, lcce_type
                )

                backward_useful_causal_event_list, _ = search_on_direction(
                    args, current_root_cause_event, "backward", history_candidates,
                    causal_graph, events, modified_all_event, question, log_answer,
                    output_dir, file_basename, ground_truth, lcce_client, lcce_type
                )

                # 
                useful_causal_event_list.update(forward_useful_causal_event_list)
                useful_causal_event_list.update(backward_useful_causal_event_list)
                
                
                final_mistake_info = identify_final_mistake(
                    useful_causal_event_list, events, final_mistake_info_path, question, all_events_causality, lcce_client, ground_truth, lcce_type, args.max_retries
                )
            
            final_root_cause_event, final_root_reason = final_mistake_info["root_cause_event"], final_mistake_info["reason"]
            logger.info(f"\n--- Final Root Cause for {file_basename}: {final_root_cause_event} ---")
            
            # # Decompose root_cause_event and write to file
            
            final_event_step = final_root_cause_event.split('_')[1]  # Assuming event_id format is "event_<step>_<id>"
            final_mistake_agent = log_content[int(final_event_step)].get('role', 'Unknown').split(' ')[0] if "Hand-Crafted" in args.input_dir else log_content[int(final_event_step)].get('name', 'Unknown')

            with open(args.output, "a", encoding="utf-8") as f:
                f.write(f"Prediction for {os.path.basename(log_file)}:\n")
                f.write(f"Agent Name: {final_mistake_agent}  \n")
                f.write(f"Step Number: {final_event_step}  \n")
                f.write(f"Reason for Mistake: {final_root_reason}\n\n")
                f.write("==================================================\n\n")
            
            with open(history_str_path, "a", encoding="utf-8") as f:
                f.write("==================================================\n\n")
                f.write(f"Prediction for {os.path.basename(log_file)}:\n")
                f.write(f"Agent Name: {final_mistake_agent}  \n")
                f.write(f"Step Number: {final_event_step}  \n")
                f.write(f"Reason for Mistake: {final_root_reason}\n\n")
                f.write("==================================================\n\n")
            
            logger.info(f"Agent Name: {final_mistake_agent} Step Number: {final_event_step} Reason for Mistake: {final_root_reason}")
        
        except Exception as e:
            logger.error(f"An error occurred during processing of {log_file}: {e}. Skipping to next log file.")
            final_mistake_agent = log_content[int(final_event_step)].get('role', 'Unknown').split(' ')[0] if "Hand-Crafted" in args.input_dir else log_content[int(final_event_step)].get('name', 'Unknown')
            
            with open(args.output, "a", encoding="utf-8") as f:
                f.write(f"Prediction for {os.path.basename(log_file)}:\n")
                f.write(f"Agent Name: {final_mistake_agent}  \n")
                f.write(f"Step Number: {final_event_step}  \n")
                f.write(f"Reason for Mistake: {final_root_reason}\n\n")
                f.write("==================================================\n\n")
            
            with open(history_str_path, "a", encoding="utf-8") as f:
                f.write("==================================================\n\n")
                f.write(f"Prediction for {os.path.basename(log_file)}:\n")
                f.write(f"Agent Name: {final_mistake_agent}  \n")
                f.write(f"Step Number: {final_event_step}  \n")
                f.write(f"Reason for Mistake: {final_root_reason}\n\n")
                f.write("==================================================\n\n")
            continue
    logger.info(f"\nFinal aggregated attribution results saved to {args.output}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Causal Mediation Analysis for Agent Failure Attribution")
    
    # Core I/O arguments
    parser.add_argument("--input_dir", type=str, default=os.path.join(parent_dir, "Who&When/Algorithm-Generated"), help="Algorithm-Generated or Hand-Crafted")
    parser.add_argument("--output", type=str, default=os.path.join(current_dir, "outputs/attribution_result.json"), help="Final aggregated attribution result file (.txt)")
    
    # Model and execution parameters
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-pro", choices=ALL_MODELS,
        help=f"Model identifier. Choose from: {', '.join(ALL_MODELS)}"
    )
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API Key. Uses OPENAI_API_KEY env var if not set.")
    parser.add_argument("--base_url", type=str, default=os.getenv("OPENAI_BASE_URL"), help="OpenAI compatible base URL. Uses OPENAI_BASE_URL env var if not set.")
    parser.add_argument(
        "--LCCE", type=str, default="qwen3-coder-30b-a3b-instruct", choices=LOCAL_MODEL_ALIASES,
        help=f"LCCE. Choose from: {', '.join(LOCAL_MODEL_ALIASES)}"
    )
    parser.add_argument(
        "--local_llm_type", type=str, default="vllm", choices=["local", "vllm"],
        help="Type of local LLM server to use."
    )
    parser.add_argument(
        "--vllm_api_key", type=str, default=os.getenv("VLLM_API_KEY"), help="API key for local LLM server.",
    )
    parser.add_argument(
        "--vllm_base_url", type=str, default=os.getenv("VLLM_BASE_URL"), help="Base URL for local LLM server.",
    )
    parser.add_argument(
        "--local_model_path", type=str, default="<path>",
        help=f"local model path for LCCE models"
    )
    parser.add_argument(
        "--device", type=str, default=get_default_device(),
        help="Device for local model inference (e.g., 'cuda:0', 'mps', 'cpu')."
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Window size for inter-agent causality")
    parser.add_argument("--max_iterations", type=int, default=5, help="Max iterations for root cause search")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries for a failed LLM call")

    args = parser.parse_args()

    main(args)
