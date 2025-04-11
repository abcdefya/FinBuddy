import os
from marker.convert import convert_single_pdf
from marker.models import load_all_models
from marker.output import save_markdown

SAVE_DIR = "output/SEC_EDGAR_FILINGS_MD"

def run_marker(
        input_ticker_year_path: str, output_ticker_year_path: str, batch_multiplier: int = 2):

    model_lst = load_all_models()
    for input_path in os.listdir(input_ticker_year_path):
        if not input_path.endswith(".pdf"):
            continue
        input_path_full = os.path.join(input_ticker_year_path, input_path)
        full_text, images, out_meta = convert_single_pdf(
            input_path_full,
            model_lst,
            langs=["English"],
            batch_multiplier=batch_multiplier,
        )
        fname = os.path.basename(input_path)
        subfolder_path = save_markdown(
            output_ticker_year_path, fname, full_text, images, out_meta)
        print(f"Saved to {subfolder_path}")
    del model_lst