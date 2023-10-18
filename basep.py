import openai
import pandas as pd
from tqdm import tqdm
import os
import json
import csv
import logging

openai.api_base = ""
openai.api_key = ""

templates = {
    1: 'In the above code snippet, check for potential security vulnerabilities and output either \'Vulnerable\' or \'Non-vulnerable\'. '
       'You are now an excellent programmer.'
       'You are conducting a function vulnerability detection task for C/C++ language.',
    2: 'The node information of the function is as follows:',
    3: 'The edge information of the function is as follows:',
    4: 'Here is an example for you to learn from:'
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

fh = logging.FileHandler('devignmetricsgpt4.log')
fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
fh.setFormatter(logging.Formatter(fmt))

logger.addHandler(fh)


def main():
    with open('F:/pycharmfile/vulllm/devign_data/devign_test_processed.json', 'r') as f:
        data = json.load(f)

    def calculate_metrics(predictions, ground_truth):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0

        for pred, target in zip(predictions, ground_truth):
            if pred == target == 1:
                true_positives += 1
            elif pred == target == 0:
                true_negatives += 1
            elif pred == 1 and target == 0:
                false_positives += 1
            elif pred == 0 and target == 1:
                false_negatives += 1

        accuracy = (true_positives + true_negatives) / len(predictions)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return accuracy, precision, recall, f1

    prediction_ls = []
    ground_truth = []

    for row in data[0:2000]:
        if 'func' in row:
            inputCode = row['func'][:4000]
        if 'node' in row:
            inputnode = row['node'][:2000]
        if 'edge' in row:
            inputedge = row['edge'][:2000]
        if 'func' in row:
            inputex = row['example'][:4000]


            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": format(inputCode)+templates[1]}
                ]
            )
            prediction = response['choices'][0]['message']['content']
            print(prediction)

            with open('devignresultsgpt4.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['result'])
                writer.writerow(prediction)
                f.close()

            if prediction == "0" or prediction == "1":
                prediction = int(prediction)
            else:
                prediction = 2

            prediction_ls.append(prediction)
            ground_truth.append(row['target'])
            # print(inputCode)
            # print(inputnode)
            # print(inputedge)

        with open('devignresultsgpt4.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Prediction', 'Groundtruth'])
            writer.writerows(zip(prediction_ls, ground_truth))

        print(prediction_ls)
        print(ground_truth)

        accuracy, precision, recall, f1 = calculate_metrics(prediction_ls, ground_truth)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        logger.info("Accuracy: %f", accuracy)
        logger.info("Precision: %f", precision)
        logger.info("Recall: %f", recall)
        logger.info("F1 Score: %f", f1)

if __name__ == '__main__':
    main()


