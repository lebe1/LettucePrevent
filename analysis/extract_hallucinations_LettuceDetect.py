from lettucedetect.models.inference import HallucinationDetector
from datetime import datetime
import json


def process_summary_item(item, detector):
    if item.get("task_type") != "Summary":
        return None

    contexts = [item.get("prompt")]
    answer = item.get("answer")
    predictions = detector.predict(context=contexts, answer=answer, output_format="spans")

    if predictions:
        print("Hallucinatios:", predictions)
        item["hallucinations_detected"] = predictions
        return item
    else:
        return None

def main(input_path, json_output_path, stats_output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    detector = HallucinationDetector(
            method="transformer",
            model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1",
        )
    
    summary_items = 0
    data2txt_items = 0
    qa_items = 0
    hallucinated_items = []

    for i, item in enumerate(data):
        task_type = item.get("task_type")
        if task_type == "Summary":
            summary_items += 1
            result = process_summary_item(item, detector)
            if result:
                result["prompt_number"] = i
                hallucinated_items.append(result)
        elif task_type == "Data2txt":
            data2txt_items += 1
        elif task_type == "QA":
            qa_items += 1
        else:
            print("Incorrect task type: ", task_type)

    total_items = summary_items + data2txt_items + qa_items

    # Save hallucinated items to JSON
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(hallucinated_items, f, indent=4, ensure_ascii=False)

    # Save stats to TXT file
    with open(stats_output_path, "w", encoding="utf-8") as f:
        f.write(f"Summary entries: {summary_items}\n")
        f.write(f"Data2txt entries: {data2txt_items}\n")
        f.write(f"QA entries : {qa_items}\n")
        f.write(f"Total entries : {total_items}\n")
        f.write(f"Total entries with hallucinated numbers: {len(hallucinated_items)}\n")

    # Also print to console
    print(f"Summary entries: {summary_items}")
    print(f"Data2txt entries: {data2txt_items}")
    print(f"QA entries : {qa_items}")
    print(f"Total entries : {total_items}")
    print(f"Total entries with hallucinations in Summary: {len(hallucinated_items)}")



if __name__ == "__main__":
    input_file = "../data/summary_experiments_run_20250922_155006_number_detector_logits_processor_comparison_without.json"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_output_file = f"../data/hallucinations_experiments_LettuceDetect_{timestamp}.json"
    stats_output_file = f"../data/hallucination_stats_LettuceDetect_{timestamp}.txt"
    main(input_file, json_output_file, stats_output_file)
