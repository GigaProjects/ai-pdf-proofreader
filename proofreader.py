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
CHUNK_LENGTH = 3000      # Detailed chunk length (approx characters)
PROOFREAD_LANGUAGE = "English"
MAX_PARALLEL_REQUESTS = 20 # Parallel workers for Fast Scan (Flash)
MAX_PARALLEL_SMART = 5     # Parallel workers for Smart Verify (Pro) - keep low for rate limits
VERIFICATION_BATCH_SIZE = 20 # Number of corrections to verify at once per request
ENABLE_VERIFICATION = False # Set to False to skip the second pass and save costs

# --- MODEL CONFIGURATION ---
MODEL_FAST = "gemini-flash-latest"   # For initial proofreading (Speed/Cost)
MODEL_SMART = "gemini-flash-latest"       # For verification (Accuracy) use "gemini-2.5-pro" for best results

# --- DYNAMIC PROMPT EXAMPLES ---
# Default to English examples for simplicity
curr_ex = {
    "orig": "This is example statement.",
    "corr": "This is an example statement.",
    "word1": "work-"
}

SYSTEM_PROMPT_FLASH = f"""You are an expert {PROOFREAD_LANGUAGE} proofreader. 
Your task is to read the provided text and identify grammatical, spelling, and stylistic errors.
For each error found, provide the:
1. 'original_sentence': The complete sentence containing the error.
2. 'correction': The corrected version of the same sentence.

Return the result as a JSON object with a list under the key "corrections". 
If no errors are found, return an empty list.
Example format:
{{
  "corrections": [
    {{
      "original_sentence": "{curr_ex['orig']}",
      "correction": "{curr_ex['corr']}"
    }}
  ]
}}

IMPORTANT: Some lines in the input might be broken by hyphens (e.g. 'work-' followed by 'flow'). Treat these as a single word. 
Also, ignore errors that are obviously formatting artifacts from PDF extraction (like page numbers or single letters split from words).

The text is processed in chunks, so sentences will often start mid-sentence with a lowercase letter or end abruptly without a period. 
DO NOT flag these as errors. Only correct legitimate grammar and spelling mistakes.
If a sentence is correct despite formatting oddities, do not flag it.
"""

SYSTEM_PROMPT_PRO = f"""
You are a quality control assistant for a proofreading tool.
Your task is to review a list of suggested corrections and FILTER OUT false positives.

A "False Positive" is a correction that:
1. Is caused purely by PDF parsing errors (e.g., a word split by a newline '{curr_ex['word1']} \\n' -> '{curr_ex['word1'][:-1]}' which looks like a space insertion but isn't a grammar error).
2. Is changing a number or page number that shouldn't be there (e.g. removing a stray '23').
3. Is "correcting" a sentence that was actually cut off or formatting garbage.
4. IS CORRECTING A LOWERCASE START OR MISSING PERIOD caused by chunk splitting. Chunks start/end mid-sentence; ignore these.

KEEP only corrections that are legitimate spelling, grammar, or stylistic errors.

IMPORTANT: You must return the JSON objects for the valid items EXACTLY as they appear in the input. Do not modify the text or invent new corrections.

Return a JSON object with a single key "verified_corrections" containing ONLY the valid items from the input list.
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
        # 1. Merge hyphenated words and handling soft hyphens
        text = re.sub(r'(\w)\s*[-\xad]\s*\n\s*(\w)', r'\1\2', text)
        text = re.sub(r'(\w)\s*[-\xad]\s+(\w)', r'\1\2', text)
        
        # 2. Heuristic for page numbers and headers/footers
        lines = text.split('\n')
        cleaned_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip lines that look like page numbers or headers
            # 1. Just digits
            if stripped.isdigit():
                continue
            
            # 2. "Page X" or "X / Y" format
            # Check for specific patterns rather than just "any digit"
            if re.match(r'^(page|stran)\s*\d+$', stripped.lower()) or re.match(r'^\d+\s*/\s*\d+$', stripped):
                continue

            # 3. Very short lines with digits that aren't sentences (e.g. "2023", "Unit 1")
            # Only remove if it doesn't end with sentence punctuation
            if len(stripped) < 10 and any(c.isdigit() for c in stripped):
                if not stripped.endswith(('.', '!', '?', ':', ';')):
                     continue
                # If it's short and has numbers, it's suspicious. Check if it's NOT a sentence.
                if not stripped.endswith(('.', '!', '?')):
                    continue

            cleaned_lines.append(line)
        
        text = "\n".join(cleaned_lines)

        # 3. Aggressive "Sticky Number" Removal
        # Remove digits attached to the end of words (e.g., "beseda123" -> "beseda")
        # We assume specific Slovenian lowercase letters to avoid stripping real citations like "Model T5"
        text = re.sub(r'([a-zčšž])\d+\b', r'\1', text) 
        
        # Remove digits at the start of words (e.g., "123beseda" -> "beseda")
        text = re.sub(r'\b\d+([a-zčšž])', r'\1', text)

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

    @staticmethod
    def _call_with_retry(func, *args, retries=3, **kwargs):
        """Helper to retry API calls with exponential backoff."""
        import time
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == retries - 1: raise e
                time.sleep(2 ** attempt) # Exponential backoff: 1s, 2s, 4s

    def proofread_chunk(self, chunk):
        """Sends a single chunk to Gemini and returns corrections with page info."""
        try:
            # Wrap API call with retry logic
            response = self._call_with_retry(
                self.client.models.generate_content,
                model=MODEL_FAST,
                contents=f"{SYSTEM_PROMPT_FLASH}\n\nText to proofread:\n{chunk['text']}",
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            result = json.loads(response.text)
            corrections = result.get("corrections", [])
            for c in corrections:
                c["page"] = chunk["pages"][0]
            return corrections
        except Exception as e:
            # Return empty on persistent failure
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
        
        # Define base_name and timestamp before using them
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]

        if all_corrections:
            if ENABLE_VERIFICATION:
                # 1. Save Raw Results First (Only if we are going to verify afterward)
                raw_output_file = f"results_{base_name}_{timestamp}_RAW.txt"
                console.print(f"[yellow]Saving raw results to: {raw_output_file}...[/yellow]")
                self.display_and_save_results(all_corrections, raw_output_file, show_console=False)

                # 2. Verify with Smart Model (with progress bar)
                final_results = self.verify_corrections(all_corrections)
                final_suffix = "VERIFIED"
            else:
                # Skip verification and use raw corrections directly
                console.print("[blue]Skipping Smart Verification (ENABLE_VERIFICATION is False)[/blue]")
                final_results = all_corrections
                final_suffix = "FAST"
            
            # Sort final results
            final_results.sort(key=lambda x: x.get("page", 0))

            # 3. Save Final Results
            final_output_file = f"results_{base_name}_{timestamp}_{final_suffix}.txt"
            self.display_and_save_results(final_results, final_output_file, show_console=True)
            
        else:
             console.print("[green]No errors found![/green]")

    def verify_corrections(self, corrections):
        """
        Sends the list of corrections back to the LLM in parallel to filter out false positives.
        """
        verified_corrections = []
        total_items = len(corrections)
        
        # Create batches
        batches = [corrections[i:i+VERIFICATION_BATCH_SIZE] for i in range(0, total_items, VERIFICATION_BATCH_SIZE)]
        
        def process_batch(batch):
            prompt = f"""{SYSTEM_PROMPT_PRO}
            
            Input JSON list:
            {json.dumps(batch, ensure_ascii=False)}
            """
            try:
                # Wrap API call with retry logic
                response = self._call_with_retry(
                    self.client.models.generate_content,
                    model=MODEL_SMART,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                result = json.loads(response.text)
                return result.get("verified_corrections", [])
            except Exception as e:
                console.print(f"[yellow]Warning: Verification batch failed: {e}. Keeping original items.[/yellow]")
                return batch

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Smart Verification ({MODEL_SMART})...", total=total_items)
            
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SMART) as executor:
                futures = {executor.submit(process_batch, b): b for b in batches}
                
                for future in as_completed(futures):
                    batch_res = future.result()
                    verified_corrections.extend(batch_res)
                    progress.update(task, advance=len(futures[future]))
                
        return verified_corrections

    def display_and_save_results(self, corrections, output_file, show_console=True):
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

        # Print summary to console
        if show_console:
            # console.print(table) # Uncomment if you want to see the full table for verified results
            pass
        
        console.print(f"[bold yellow]Results saved to:[/bold yellow] {output_file} ([bold cyan]{len(corrections)} errors[/bold cyan])")

if __name__ == "__main__":
    import sys
    target_file = sys.argv[1] if len(sys.argv) > 1 else "document.pdf"
    
    if not os.path.exists(target_file):
        console.print(f"[yellow]File '{target_file}' not found.[/yellow]")
        target_file = console.input("Enter path to PDF file: ").strip().strip('"')

    tool = ProofreaderTool()
    tool.run(target_file)
