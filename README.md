# ai-pdf-proofreader

High accuracy PDF proofreading tool using Gemini AI. Features automated text cleaning, parallel processing, and multi-language support.

## Features

- Extracts text from PDF files.
- Cleans PDF artifacts like hyphenated line breaks and orphaned page numbers.
- Processes text in parallel for high speed.
- Supports configurable languages.
- Exports results to a timestamped file.

## Installation

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the root directory and add your Gemini API key:
   ```text
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

Run the script by providing the path to a PDF file:
```bash
python proofreader.py document.pdf
```
If no file is provided as an argument, the script will prompt you for a path.

## Configuration

You can modify several variables at the top of `proofreader.py`:

- `PROOFREAD_LANGUAGE`: Set the target language (default is "English").
- `MAX_PARALLEL_REQUESTS`: Number of simultaneous API calls (default is 20).
- `CHUNK_LENGTH`: Approximate number of characters per processing chunk.
- `MODEL_SELECTED`: The Gemini model version to use.

## Output

The tool provides a table in the console showing:
- Page number
- Original sentence
- Corrected sentence

The same results are automatically saved to a file named `results_[filename]_[timestamp].txt`.


