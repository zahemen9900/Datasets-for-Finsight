# Datasets-for-Finsight
A curation of the datasets used for adapting (fine-tuning) SmolLM-V2 to Finsight-AI, v1

## Project Overview

This repository contains the curated datasets used for fine-tuning SmolLM2-1.7B-Instruct model to create Finsight-AI, a specialized financial domain assistant. The fine-tuning was performed using QLoRA (Quantized Low-Rank Adaptation) to enhance the model's performance in financial contexts while maintaining computational efficiency.

A more detailed paper on the project can be found in this repo here: [Model Paper](metrics/model_paper.md)

## Dataset Composition

The dataset consists of 10,896 conversations containing approximately 16.5 million tokens, with an average of 1,521.6 tokens per conversation. The dataset is composed of five main categories:

1. **Financial Definitions**: Structured explanations of financial terms and concepts, curated from financial PDFs and enhanced with distilled responses from GPT-4o, Claude 3.5/3.7 - Sonnet, and Gemini 2.0 Flash.

2. **Financial Conversations**: Multi-turn dialogues covering investment advice, market analysis, and financial planning, synthetically generated using high-quality distilled responses from LLMs.

3. **Company Q&A**: Questions and answers about specific companies, earnings reports, and financial statements, sourced from a Finance Q&A dataset. 

4. **Introduction Conversations**: Opening dialogues establishing financial advisory context, synthetically generated with the identity of "FinSight", a financial advisor engineered to assist users with financial queries.

5. **Reddit Finance Comments**: Heavily filtered comments from the Reddit-250K-Dataset to adapt a more conversational and informal tone.

### Dataset Statistics

| Dataset               | Total Tokens | Conversations | Avg Tokens/Conv | System Avg | User Avg | Assistant Avg | 95th %ile |
|-----------------------|--------------|---------------|-----------------|------------|----------|---------------|-----------|
| Financial Introductions | 381,703      | 1,000         | 381.7           | 26.2       | 31.8     | 86.7          | 428       |
| Reddit Finance        | 6,994,396    | 4,542         | 1,539.9         | 31.8       | 80.2     | 145.8         | 2,429     |
| Company-Specific Q&A   | 1,010,581    | 1,354         | 746.4           | 32.2       | 45.4     | 67.5          | 1,145     |
| Financial Definitions | 1,167,424    | 2,000         | 583.7           | 26.0       | 37.4     | 102.6         | 1,282     |
| Finance Conversations | 7,025,430    | 2,000         | 3,512.7         | 26.1       | 37.5     | 480.7         | 5,670     |



> The method used to generate these datasets can be found in the main repo: [Findight-AI](https://github.com/zahemen9900/FinsightAI.git)

> Links to publicly available datasets can also be found here: [Reddit Finance](https://huggingface.co/datasets/winddude/reddit_finance_43_250k), [Company Q&A](https://huggingface.co/datasets/virattt/financial-qa-10K)

## Fine-tuning Approach

The datasets were used to fine-tune SmolLM2-1.7B-Instruct using QLoRA with the following key configurations:

- **Rank (r)**: 64
- **Alpha**: 16
- **Dropout**: 0.05
- **Quantization**: 4-bit NormalFloat (NF4)
- **Training**: 2 epochs on consumer-grade hardware (NVIDIA RTX 3050 GPU)

## Results

The fine-tuned model demonstrated significant improvements across all evaluation metrics:

| Metric | Base Model | QLora Model | Improvement % |
|--------|------------|-------------|---------------|
| rouge1 | 0.1777     | 0.1962      | 10.37%        |
| rouge2 | 0.0185     | 0.0264      | 42.93%        |
| rougeL | 0.0896     | 0.1057      | 17.94%        |
| bleu   | 0.0054     | 0.0091      | 68.43%        |

## Repository Structure

- `/datasets`: Contains the processed datasets used for fine-tuning
  - `/financial_definitions`: Financial terms and concepts
  - `/financial_conversations`: Multi-turn dialogues on financial topics
  - `/company_qa`: Company-specific questions and answers
  - `/introductions`: Opening dialogues for financial advisory context
  - `/reddit_finance`: Filtered Reddit finance comments

- `/metrics`: Contains research papers and evaluation results
  - `research_paper.md`: Detailed research methodology and findings

- `/visualizations`: Charts and graphs showing model performance

## Usage

To use these datasets for fine-tuning your own models:

1. Clone this repository
2. Install the required dependencies
3. Follow the fine-tuning scripts in the `/scripts` directory

## Citation

If you use these datasets in your research, please cite:

```
@article{finsight2025,
  title={FinSight AI: Enhancing Financial Domain Performance of Small Language Models Through QLoRA Fine-tuning},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2025}
}
```

## License

[Specify license information]

## Acknowledgments

Special thanks to the developers of the SmolLM2 model family for creating an accessible base model that enabled this research.
