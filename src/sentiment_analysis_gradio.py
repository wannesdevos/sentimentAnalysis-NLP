import gradio as gr
from models import predict_model_1, predict_model_2


def analyze_sentiment(text, model_choice):
    """Analyze sentiment using the selected model."""
    if not text.strip():
        return "Please enter some text", 0.0

    if model_choice == "Model 1 - DistilBERT (Fast)":
        label, score = predict_model_1(text)
    else:  # Model 2 - RoBERTa Large (Accurate)
        label, score = predict_model_2(text)

    return label.capitalize(), score


# Create Gradio interface
with gr.Blocks(title="Sentiment Analysis") as demo:
    gr.Markdown("# Sentiment Analysis Demo")
    gr.Markdown("Enter text to analyze its sentiment (positive or negative)")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter your text",
                placeholder="Type a movie review, comment, or any text...",
                lines=5,
            )
            model_dropdown = gr.Dropdown(
                choices=[
                    "Model 1 - DistilBERT (Fast)",
                    "Model 2 - RoBERTa Large (Accurate)",
                ],
                value="Model 1 - DistilBERT (Fast)",
                label="Select Model",
            )
            analyze_btn = gr.Button("Analyze Sentiment", variant="primary")

        with gr.Column():
            sentiment_output = gr.Textbox(label="Sentiment", interactive=False)
            confidence_output = gr.Number(label="Confidence Score", interactive=False)

    # Examples
    gr.Examples(
        examples=[
            [
                "This movie was absolutely fantastic! Best film I've seen this year.",
                "Model 1 - DistilBERT (Fast)",
            ],
            ["Terrible movie, waste of time and money.", "Model 1 - DistilBERT (Fast)"],
            [
                "The acting was decent but the plot was confusing.",
                "Model 2 - RoBERTa Large (Accurate)",
            ],
        ],
        inputs=[text_input, model_dropdown],
    )

    # Connect the button to the function
    analyze_btn.click(
        fn=analyze_sentiment,
        inputs=[text_input, model_dropdown],
        outputs=[sentiment_output, confidence_output],
    )


if __name__ == "__main__":
    demo.launch()
