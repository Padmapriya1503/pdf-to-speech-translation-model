import fitz  # PyMuPDF
from transformers import VitsModel, MBartForConditionalGeneration, AutoTokenizer
import torch
import soundfile as sf
import gradio as gr

# Load the translation model and tokenizer
translation_tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", use_fast=False)
translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")

# Load the TTS model and tokenizer
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
tts_model = VitsModel.from_pretrained("facebook/mms-tts-hin")

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def process_pdf(pdf_file):
    # Extract text from the PDF
    input_text = extract_text_from_pdf(pdf_file)

    # Convert sentences to tensors
    model_inputs = translation_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Translate from English to Hindi
    generated_tokens = translation_model.generate(
        **model_inputs,
        forced_bos_token_id=translation_tokenizer.lang_code_to_id["hi_IN"]
    )

    # Decode the translated tokens to text
    translation = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    translated_text = " ".join(translation)  # Join all translated sentences

    # Tokenize the translated text for TTS
    tts_inputs = tts_tokenizer(translated_text, return_tensors="pt")

    # Generate the waveform
    try:
        with torch.no_grad():
            tts_output = tts_model(**tts_inputs)
            waveform = tts_output.waveform.squeeze().cpu().numpy()
    except RuntimeError as e:
        return f"Runtime Error: {e}"

    # Save the waveform to an audio file
    audio_path = "output.wav"
    sf.write(audio_path, waveform, 22050)

    return audio_path

def gradio_interface(pdf_file):
    audio_path = process_pdf(pdf_file.name)
    return audio_path

# Create the Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.File(file_count="single"),
    outputs="audio"
)

# Launch the Gradio app
iface.launch(debug=True)
