import re

# Read the content of the predictions file
with open('/home/eli/Downloads/gmu work/runs/person/7_23_person/test_predictions_simple.txt', 'r') as f:
    content = f.read()

mistakes_output = []

# Split the content into individual examples
examples = re.split(r'={10,}', content)

for example in examples:
    if not example.strip():
        continue

    # Extract GROUND TRUTH and PREDICTED lines
    gt_match = re.search(r'GROUND TRUTH:\n\["(.*?)"\]', example)
    pred_match = re.search(r'PREDICTED:\n\["(.*?)"\]', example)

    if gt_match and pred_match:
        ground_truth_str = gt_match.group(1)
        predicted_str = pred_match.group(1)

        # Convert comma-separated strings to sets for comparison
        ground_truth_set = set(gt.strip() for gt in ground_truth_str.split('", "')) if ground_truth_str else set()
        predicted_set = set(pred.strip() for pred in predicted_str.split('", "')) if predicted_str else set()

        # Compare the sets
        if ground_truth_set != predicted_set:
            mistakes_output.append(example.strip())

# Join the mistakes and write to a new file
output_content = "\n====================\n\n".join(mistakes_output)

with open('/home/eli/Downloads/gmu work/runs/person/7_23_newdata/mistakes_predictions.txt', 'w') as f:
    f.write(output_content)

print("Successfully created mistakes_predictions.txt")