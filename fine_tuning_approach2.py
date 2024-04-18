import csv
import json

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
    with open("train_10_samples_approach2.csv", "r") as csv_file, open("train_10_samples_approach2.jsonl", "w") as jsonl_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            prompt = row["prompt"]
            flashcards = []
            for i in range(1, 6):
                front_key = f"front{i}"
                back_key = f"back{i}"
                if row[front_key] and row[back_key]:
                    flashcard = {"front": row[front_key], "back": row[back_key]}
                    flashcards.append(flashcard)
            example = create_dataset(prompt, flashcards)
            jsonl_file.write(json.dumps(example) + "\n")