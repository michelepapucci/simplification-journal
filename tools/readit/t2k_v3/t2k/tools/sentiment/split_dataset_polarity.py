import re


def process_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_file, 'w', encoding='utf-8') as file:
        for line in lines:
            if line.strip().startswith('<doc') and 'task_polarity' in line:
                # Extract task_polarity value
                task_polarity = re.search(r'task_polarity="([^"]+)"', line).group(1)

                # Determine task_positive and task_negative values
                task_positive, task_negative = '0', '0'
                if task_polarity == 'NEG':
                    task_negative = '1'
                elif task_polarity == 'POS':
                    task_positive = '1'
                elif task_polarity == 'POS_NEG':
                    task_positive, task_negative = '1', '1'

                # Add task_positive and task_negative to the line
                line = line.strip()[:-1] + f' task_positive="{task_positive}" task_negative="{task_negative}"' + \
                       line.strip()[-1] + '\n'

            file.write(line)


if __name__ == "__main__":
    input_file = 'training/training_sentiment.pos'
    output_file = 'training/training_sentiment_splitted_polarity.conllu'
    process_dataset(input_file, output_file)
