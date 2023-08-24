import re
from tabulate import tabulate

# summed_dataPath = './RsRecords/td_0811saved.txt'
summed_dataPath = "./RsRecords/t_PartSum0813.txt"

# Define a function to extract data from the content of the file
def extract_data(content):
    pattern = (r"saveDate:\s*(\S+)"
               r".*?Ansatz:\s*(\S+)"
               r".*?NUM_LAYERS:\s*(\d+)"
               r".*?Num_Epoch:\s*(\d+)"
               r".*?SG_Weight:\s*(\d+)"
               r".*?Optimizer:\s*(\S+)"
               r".*?Step_Size:\s*(\S+)"
               r".*?bestTrainAcc:\s*(\S+),\s*bestTrainAccTest:\s*(\S+),\s*at\s*Epoch:\s*(\d+)"
               r".*?bestTrainAccG:\s*(\S+),\s*bestTrainAccGTest:\s*(\S+),\s*at\s*Epoch:\s*(\d+)")

    return re.findall(pattern, content, re.DOTALL)


# Read the file
with open(summed_dataPath, "r") as file:
    content = file.read()

# Extract data
data = extract_data(content)
raw_header = ["saveDate", "Ansatz", "NUM_LAYERS", "Num_Epoch", "SG_Weight", "Optimizer", "Step_Size",
              "bestTrainAcc", "bestTrainAccTest", "Epoch", "bestTrainAccG", "bestTrainAccGTest", "EpochG"]
data_dicts = [dict(zip(raw_header, row)) for row in data]

sorted_rs = sorted(data_dicts, key=lambda x: (
    x["Ansatz"], x["Optimizer"], int(x["NUM_LAYERS"]), float(x["SG_Weight"]), float(x["Step_Size"])))

present_header = ["saveDate", "Num_Epoch", "Ansatz", "Optimizer", "NUM_LAYERS", "SG_Weight",  "Step_Size",
                  "bestTrainAcc", "bestTrainAccG", "bestTrainAccTest", "bestTrainAccGTest", "Epoch", "EpochG"]

# Create a table header

present_rs = [[entry[key] for key in present_header] for entry in sorted_rs]
print(tabulate(present_rs, present_header, tablefmt='pretty'))

# not used: The Final rs
# r".*?acc_train:\s*(\S+),\s*acc_test:\s*(\S+)"
# r".*?accG_train:\s*(\S+),\s*accG_test:\s*(\S+)"
