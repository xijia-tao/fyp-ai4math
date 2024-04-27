# Large Language Models for Formal Theorem Proving in Lean

This README provides instructions for setting up, training, and evaluating large language models (LLMs) specifically tailored for formal theorem proving using the Lean theorem prover. Follow these steps to replicate our development environment and experiment settings.

## Environment Setup

1. Install the required dependencies via the provided `requirements.txt` file using the following command:
   ```sh
   pip install -r requirements.txt
   ```
   Alternatively, ensure you have the compatible dependencies installed manually.

## Fine-Tuning Dataset Preparation

1. Download the **MUSTARDSauce** dataset from the official [repository](https://github.com/Eleanor-H/MUSTARD) and unzip it into the `data` directory:
   ```plaintext
   data/MUSTARDSauce
   ```
2. Download the **MMA** data from the official [repository](https://github.com/albertqjiang/MMA) and place it into the following directory:
   ```plaintext
   data/MMA
   ```
3. Execute the `prep/process.py` script to process the datasets:
   ```sh
   python prep/process.py
   ```

## Train Large Language Models (LLMs)

1. We have trained **Mistral 7B** and **CodeLLaMa 7B** models, but you can fine-tune other models by modifying the `MODEL_NAME` parameter within the relevant Python scripts.
   
2. Run the script `train/train_mustard.py` to fine-tune the base models. Adjust the settings within this file to accommodate different experimental parameters.

3. The script `train/train_mma.py` is specifically for the autoformalization task. Though not directly involved in our final experiments, it can be used for related tasks.

## Evaluation Setup

1. Clone the **miniF2F** [repository](https://github.com/openai/miniF2F) under the root directory as `minif2f`.
   
2. Transfer the `eval/verify.py` script into the `minif2f/lean` directory.
   
3. Set up the Lean environment according to the Lean specifications provided in the **minif2f** repository.

4. Modify the data fields (e.g., the path to the directory containing inference results) in the Python scripts contained in the `eval/` directory to reflect your environment.

5. To conduct the evaluation, execute the corresponding Python scripts within `eval/methods` for different prompting methods and local/OpenAI LLMs, and sequentially run `extract.py`, `verify.py`, and `get_results.py`:
   ```sh
   python eval/methods/<script_name>.py
   python eval/extract.py
   python eval/verify.py
   python eval/get_results.py
   ```
   The `all_results.json` file containing the final results will be generated in the designated results directory.

## Remark

Unless instructions specify otherwise, run all scripts from the root directory of the main repository.

---

## VS Code extension
`Lean3 Local Copilot` was developed as a side product for our project. You can find in the editor's Marketplace. The source code is also provided in `lean3-local-copilot`.
