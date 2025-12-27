from lettucedetect.models.inference import HallucinationDetector
from datetime import datetime
import json


def process_summary_item(item, detector):
    if item.get("task_type") != "Summary":
        return None

    contexts = [item.get("prompt")]
    answer = item.get("answer")
    predictions = detector.predict(context=contexts, answer=answer, output_format="spans")

    # Filter hallucinations with confidence >= 0.70
    filtered_predictions = [h for h in predictions if h['confidence'] >= 0.70]

    if filtered_predictions:
        print("Hallucinations:", filtered_predictions)
        item["hallucinations_detected"] = filtered_predictions
        return item
    else:
        return None


def categorize_hallucinations_by_confidence(hallucinated_items):
    """Categorize hallucinations into confidence intervals and return structured data"""
    
    confidence_buckets = {
        "95%-100%": [],
        "90%-95%": [],
        "85%-90%": [],
        "80%-85%": [],
        "75%-80%": [],
        "70%-75%": []
    }
    
    counts = {
        "95%-100%": 0,
        "90%-95%": 0,
        "85%-90%": 0,
        "80%-85%": 0,
        "75%-80%": 0,
        "70%-75%": 0
    }
    
    for item in hallucinated_items:
        for hallucination in item.get("hallucinations_detected", []):
            conf = hallucination['confidence']
            
            # Determine which bucket (exclusive lower, inclusive upper)
            if 0.95 < conf <= 1.0:
                confidence_buckets["95%-100%"].append(hallucination)
                counts["95%-100%"] += 1
            elif 0.90 < conf <= 0.95:
                confidence_buckets["90%-95%"].append(hallucination)
                counts["90%-95%"] += 1
            elif 0.85 < conf <= 0.90:
                confidence_buckets["85%-90%"].append(hallucination)
                counts["85%-90%"] += 1
            elif 0.80 < conf <= 0.85:
                confidence_buckets["80%-85%"].append(hallucination)
                counts["80%-85%"] += 1
            elif 0.75 < conf <= 0.80:
                confidence_buckets["75%-80%"].append(hallucination)
                counts["75%-80%"] += 1
            elif 0.70 < conf <= 0.75:
                confidence_buckets["70%-75%"].append(hallucination)
                counts["70%-75%"] += 1
    
    return confidence_buckets, counts


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

    # Categorize hallucinations by confidence intervals
    confidence_buckets, counts = categorize_hallucinations_by_confidence(hallucinated_items)
    
    # Calculate total hallucinations
    total_hallucinations = sum(counts.values())
    
    # Prepare JSON output structure
    json_output = []
    for interval, hallucinations in confidence_buckets.items():
        if hallucinations:  # Only include intervals that have hallucinations
            json_output.append({
                f"Confidence interval {interval}": {
                    "Hallucinations": hallucinations
                }
            })
    
    # Save hallucinations categorized by confidence to JSON
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=4, ensure_ascii=False)

    # Save stats to TXT file
    with open(stats_output_path, "w", encoding="utf-8") as f:
        f.write(f"Summary entries: {summary_items}\n")
        f.write(f"Data2txt entries: {data2txt_items}\n")
        f.write(f"QA entries: {qa_items}\n")
        f.write(f"Total entries: {total_items}\n")
        f.write(f"Total entries with hallucinated numbers: {len(hallucinated_items)}\n")
        f.write(f"Total hallucinations: {total_hallucinations}\n")
        f.write(f"Hallucinations in confidence interval between 95%-100%: {counts['95%-100%']}\n")
        f.write(f"Hallucinations in confidence interval between 90%-95%: {counts['90%-95%']}\n")
        f.write(f"Hallucinations in confidence interval between 85%-90%: {counts['85%-90%']}\n")
        f.write(f"Hallucinations in confidence interval between 80%-85%: {counts['80%-85%']}\n")
        f.write(f"Hallucinations in confidence interval between 75%-80%: {counts['75%-80%']}\n")
        f.write(f"Hallucinations in confidence interval between 70%-75%: {counts['70%-75%']}\n")

    # Also print to console
    print(f"\nSummary entries: {summary_items}")
    print(f"Data2txt entries: {data2txt_items}")
    print(f"QA entries: {qa_items}")
    print(f"Total entries: {total_items}")
    print(f"Total entries with hallucinations in Summary: {len(hallucinated_items)}")
    print(f"Total hallucinations: {total_hallucinations}")
    print(f"Hallucinations in confidence interval between 95%-100%: {counts['95%-100%']}")
    print(f"Hallucinations in confidence interval between 90%-95%: {counts['90%-95%']}")
    print(f"Hallucinations in confidence interval between 85%-90%: {counts['85%-90%']}")
    print(f"Hallucinations in confidence interval between 80%-85%: {counts['80%-85%']}")
    print(f"Hallucinations in confidence interval between 75%-80%: {counts['75%-80%']}")
    print(f"Hallucinations in confidence interval between 70%-75%: {counts['70%-75%']}")


if __name__ == "__main__":
    input_file = "../data/summary_experiments_tinylettuce_run_20251009_122206.json"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_output_file = f"../data/hallucinations_experiments_LettuceDetect_{timestamp}.json"
    stats_output_file = f"../data/hallucination_stats_LettuceDetect_{timestamp}.txt"
    main(input_file, json_output_file, stats_output_file)