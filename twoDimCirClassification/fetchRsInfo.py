import os
import re

folder_path = "./RsRecords"  # Replace with the path to your folder containing the .txt files
output_file_path = "./RsRecords/t_PartSum0813.txt"

def extract_info(content):
    # Define the parts that you want to extract
    patterns = [
        "saveDate:.*?\n",
        "Ansatz:.*?\n",
        "NUM_LAYERS:.*?\n",
        "Num_Epoch:.*?\n",
        "SG_Weight:.*?\n",
        "Optimizer:.*?\n",
        "Step_Size:.*?\n",
        "FinalTrain:.*?bestTrainAccG:.*?at Epoch:.*?\n"
    ]

    extracted_data = []
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            extracted_data.append(match.group())

    return "".join(extracted_data)


# Open the final output file in write mode
with open(output_file_path, 'w') as output_file:
    # Iterate over each file in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                content = file.read()
                extracted_content = extract_info(content)
                # Write the extracted content to the output file
                output_file.write(extracted_content)
                # Separate content from different files with a few line breaks
                output_file.write("\n-------------------------------------------------\n\n")

print("save finished.")
