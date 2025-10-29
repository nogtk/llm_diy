import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    verdict = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', verdict)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:100])  # Display the first 100 tokens for verification
