import json
import random

DEFAULT_SYSTEM_PROMPT = "You are an assistant that helps users create flashcards based on the content they provide. Your task is to generate a set of flashcards in JSON format, where each flashcard has a 'front' (question or term) and a 'back' (answer or definition). The number of flashcards generated should match the user's request. Aim to create concise, informative, and well-structured flashcards that effectively capture the key points of the given content."

def create_dataset(prompt, flashcards):
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(flashcards)},  # Convert flashcards to JSON string
        ]
    }

if __name__ == "__main__":
    # Read the medical flashcards dataset from the JSON file
    with open("medical_meadow_wikidoc_medical_flashcards.json", "r") as json_file:
        flashcards_data = json.load(json_file)

    # Generate the JSONL file
    with open("medical_flashcards.jsonl", "w") as jsonl_file:
        for _ in range(1, 400):  # Generate 400 samples
            n = random.randint(1, 25)  # Randomly select the number of flashcards (n) between 1 and 25
            prompt = f"Create {n} flashcards about medicine."
            flashcards = []

            # Randomly select n flashcards from the dataset
            selected_flashcards = random.sample(flashcards_data, n)

            for flashcard in selected_flashcards:
                front = flashcard["input"]
                back = flashcard["output"]
                flashcards.append({"front": front, "back": back})

            example = create_dataset(prompt, flashcards)
            jsonl_file.write(json.dumps(example) + "\n")