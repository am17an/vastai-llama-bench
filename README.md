Run `llama-bench` over various GPUs using the [vastai](https://cloud.vast.ai/) CLI

Vastai has multiple GPUs, including older architechtures like Volta and Pascal. This makes it ideal for `llama.cpp` development as one of the project's goals is broad hardware support.

## Steps 
1. Copy your patch ( for eg. `git diff upstream/master > patch.diff`) into the local folder 
2. Modify `setup_script.sh` to run the appropriate `llama-bench` command
3. Run command eg.`python3 vastai_benchmark.py --instance RTX_4090`


## TODO
- [] Add ability to specify multiple GPUs
- [] Add cost for a particular test

## Caveats
1. Sometimes, the automatic instance finding takes a long time. You can also supply an existing instance via `--instance-id`

