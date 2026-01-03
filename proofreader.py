import os
from google import genai
from google.genai import types
import PyPDF2
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich import print
from dotenv import load_dotenv
import json
import typing
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIGURATION VARIABLES ---
CHUNK_LENGTH = 3000  # Detailed chunk length (approx characters)
MODEL_SELECTED = "gemini-flash-latest" 
MAX_PARALLEL_REQUESTS = 20 # Adjust based on your API tier limits
PROOFREAD_LANGUAGE = "English"

SYSTEM_PROMPT = f"""You are an expert {PROOFREAD_LANGUAGE} proofreader. 
Your task is to read the provided text and identify grammatical, spelling, and stylistic errors.
For each error found, provide the:
1. 'original_sentence': The complete sentence containing the error.
2. 'correction': The corrected version of the same sentence.

Return the result as a JSON object with a list under the key "corrections". 
If no errors are found, return an empty list.
Example format:
{
  "corrections": [
    {
      "original_sentence": "This is example statement.",
      "correction": "This is an example statement."
    }
  ]
}

IMPORTANT: Some lines in the input might be broken by hyphens (e.g. 'work-' followed by 'flow'). Treat these as a single word. 
Also, ignore errors that are obviously formatting artifacts from PDF extraction (like page numbers or single letters split from words).
If a sentence is correct despite formatting oddities, do not flag it.
"""

# --- SETUP ---
load_dotenv()
console = Console()

class ProofreaderTool:
    def __init__(self, api_key=None):
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            console.print("[bold red]Error:[/bold red] GEMINI_API_KEY not found in environment or arguments.")
            return

        self.client = genai.Client(api_key=api_key)

    def extract_text_from_pdf(self, pdf_path):
        """Extracts text from PDF, returning a list of dictionaries with page and text."""
        console.print(f"[blue]Reading PDF:[/blue] {pdf_path}")
        extracted_content = []
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        cleaned_text = self.clean_text(text)
                        extracted_content.append({"page": i + 1, "text": cleaned_text})
            return extracted_content
        except Exception as e:
            console.print(f"[bold red]Failed to read PDF:[/bold red] {e}")
            return []

    def clean_text(self, text):
        """Clean PDF artifacts: merge hyphenated broken words and remove page numbers."""
        # 1. Merge hyphenated words at line breaks.
        # Handle formats like 'word- \n next', 'word - \n next', and 'word - next' (if newline was lost)
        # This regex looks for a word char, followed by optional spaces, a hyphen, 
        # then either a newline with optional surrounding whitespace OR at least one space.
        # We replace it with just the captured word chars.
        
        # Pattern 1: Hyphen followed by newline (standard line break)
        text = re.sub(r'(\w)\s*[-\xad]\s*\n\s*(\w)', r'\1\2', text)
        
        # Pattern 2: Hyphen followed by spaces (where newline might have been stripped by extraction)
        text = re.sub(r'(\w)\s*[-\xad]\s+(\w)', r'\1\2', text)
        
        # 2. Heuristic for page numbers: remove lines that are ONLY digits 
        # (common for headers/footers in simple PDF extractions)
        lines = text.split('\n')
        cleaned_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # If line is just a number and it's near the start or end of the page text
            if stripped.isdigit() and (i < 3 or i > len(lines) - 4):
                continue 
            cleaned_lines.append(line)
        
        text = "\n".join(cleaned_lines)

        # 3. Remove digits that got stuck to the end of the last word or start of first word
        # (e.g., 'beseda12' -> 'beseda' at the very end of a page)
        text = re.sub(r'(\w)\d+$', r'\1', text.strip()) # End of text
        text = re.sub(r'^\d+(\w)', r'\1', text)         # Start of text
            
        return text


    def chunk_text(self, pages_content):
        """Chunks text. Processes page by page for maximum precision."""
        chunks = []
        for item in pages_content:
            page_text = item["text"]
            page_num = item["page"]
            
            if len(page_text) > CHUNK_LENGTH:
                sub_chunks = [page_text[i:i+CHUNK_LENGTH] for i in range(0, len(page_text), CHUNK_LENGTH)]
                for sub in sub_chunks:
                     chunks.append({"pages": [page_num], "text": sub})
            else:
                 chunks.append({"pages": [page_num], "text": page_text})
        return chunks

    def proofread_chunk(self, chunk):
        """Sends a single chunk to Gemini and returns corrections with page info."""
        try:
            response = self.client.models.generate_content(
                model=MODEL_SELECTED,
                contents=f"{SYSTEM_PROMPT}\n\nText to proofread:\n{chunk['text']}",
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            result = json.loads(response.text)
            corrections = result.get("corrections", [])
            for c in corrections:
                c["page"] = chunk["pages"][0]
            return corrections
        except Exception as e:
            # Silently return empty on individual chunk errors to keep progress moving, 
            # though we could log it to a separate error file.
            return []

    def run(self, pdf_path):
        if not hasattr(self, 'client'): return

        pages = self.extract_text_from_pdf(pdf_path)
        if not pages: return

        chunks = self.chunk_text(pages)
        total_chunks = len(chunks)
        console.print(f"[green]Document split into {total_chunks} chunks. Starting parallel analysis...[/green]")

        all_corrections = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Proofreading...", total=total_chunks)
            
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS) as executor:
                futures = {executor.submit(self.proofread_chunk, chunk): chunk for chunk in chunks}
                
                for future in as_completed(futures):
                    result = future.result()
                    all_corrections.extend(result)
                    progress.update(task, advance=1)

        # Sort by page number
        all_corrections.sort(key=lambda x: x.get("page", 0))
        
        # Generate dynamic filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_file = f"results_{base_name}_{timestamp}.txt"
        
        self.display_and_save_results(all_corrections, output_file)

    def display_and_save_results(self, corrections, output_file):
        table = Table(title=f"{PROOFREAD_LANGUAGE} Proofreading Results: {output_file}", show_lines=True)
        table.add_column("Page", style="cyan", no_wrap=True)
        table.add_column("Original", style="red")
        table.add_column("Correction", style="green")

        # Prepare text for file saving
        file_content = f"{PROOFREAD_LANGUAGE.upper()} PROOFREADING RESULTS\n"
        file_content += "="*80 + "\n"

        for item in corrections:
            p = str(item.get("page", "?"))
            orig = item.get("original_sentence", "").strip()
            corr = item.get("correction", "").strip()
            
            table.add_row(p, orig, corr)
            file_content += f"PAGE {p}\nOLD: {orig}\nNEW: {corr}\n" + "-"*40 + "\n"

        file_content += f"\nTotal errors found: {len(corrections)}"

        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(file_content)

        # Print to console
        console.print(table)
        console.print(f"\n[bold]Total errors found:[/bold] {len(corrections)}")
        console.print(f"[bold yellow]Results also saved to:[/bold yellow] {output_file}")

if __name__ == "__main__":
    import sys
    target_file = sys.argv[1] if len(sys.argv) > 1 else "document.pdf"
    
    if not os.path.exists(target_file):
        console.print(f"[yellow]File '{target_file}' not found.[/yellow]")
        target_file = console.input("Enter path to PDF file: ").strip().strip('"')

    tool = ProofreaderTool()
    tool.run(target_file)
