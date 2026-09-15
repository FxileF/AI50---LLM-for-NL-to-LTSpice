from services.chatgpt import ChatGPTService
import asyncio
from dotenv import load_dotenv
import csv
import os

load_dotenv()

def sanitize_csv_field(value: str) -> str:
    """Remove control characters and normalize line endings for CSV export"""
    if value is None:
        return ""

    value = value.replace('\r\n', '\n').replace('\r', '\n')
    value = ''.join(
        ch for ch in value
        if ch == '\n' or ch == '\t' or ord(ch) >= 32
    )
    return value

async def main():
    """Generate circuit descriptions and ASC files using ChatGPT"""
    chat_gpt = ChatGPTService()

    tasks = [chat_gpt.generate_response(5) for _ in range(50)]
    results = await asyncio.gather(*tasks)

    file_path = 'results.csv'
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(
            file,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_ALL,
            escapechar='\\',
            lineterminator='\n'
        )

        if not file_exists:
            writer.writerow(['description', 'asc'])

        for result in results:
            for pair in result.pairs:
                writer.writerow([
                    sanitize_csv_field(pair.description),
                    sanitize_csv_field(pair.asc)
                ])

    print(f"Results appended to {file_path}")

if __name__ == "__main__":
    asyncio.run(main())
