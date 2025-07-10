from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

pdf.set_font("Arial", 'B', 16)
pdf.cell(200, 10, txt="Movie Spoiler Detector AI - Project Report", ln=True, align='C')
pdf.ln(10)

pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, txt="""
This project is a BERT-based AI system developed to detect spoilers in movie-related text content. It uses a fine-tuned BERT model to classify whether a given sentence contains a spoiler or not.

Steps Completed:

1. Data Preparation:
   - Created a dataset with labeled movie-related sentences (spoiler or not spoiler).
   - The dataset includes both spoiler and non-spoiler examples for training.

2. Model Training:
   - Used BERT (bert-base-uncased) from HuggingFace Transformers library.
   - Trained the model using a custom dataset with binary classification (0: Not Spoiler, 1: Spoiler).
   - Model and tokenizer saved to the 'spoiler_model' folder.

3. Prediction Script:
   - A command-line interface (CLI) script `run_spoiler_detector.py` lets users enter text.
   - The model returns a prediction: "Spoiler" or "Not Spoiler".

4. Evaluation:
   - Evaluated the model using a small test set.
   - Final loss: ~0.31 indicating the model is learning useful patterns.

5. Deployment:
   - Can be extended into a web app or API for integration with websites, blogs, or social media moderation tools.

Tools & Libraries Used:
- Python
- PyTorch
- HuggingFace Transformers
- BERT (bert-base-uncased)
- pandas, fpdf
""")

pdf.output("Movie_Spoiler_Detector_Report.pdf")
