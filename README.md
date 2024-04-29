# LLMCRec: LLM-empowered Course Recommendation with Multi-Grained Graph Contrastive Learning

This is our implementation for the paper:

## Environment Settings

- torch=1.10.0
- python=3.7
- tqdm=4.66.1
- numpy=1.21.6
- scipy=1.7.3

## Example to run the codes.

1. Decompress the dataset file `datasets.zip` into the current folder

2. Run the following command to train LLMCRec with GPU 0:

    ```
    python main.py -g 0 -m LLMCRec -d MOOCCubeX-CS
    ```

3. After training, you can check the log files in `./logs`

## Parameter Tuning

All the parameters are in `config.yaml`