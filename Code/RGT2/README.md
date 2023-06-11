# This is the folder for all my code and data source
## Folder Structure
* `ckpt` folder contains checkpoints saved
    * `gpt2` for standard model checkpoints 
    * `reverse-gpt2` for reverse text model checkpoints
* `config` folder contains default configuration parameters
* `data` folder contains 
    * `preprocessing` folder contains 
        > `data_preprocessing.ipynb` code for preprocessing raw data<br>
        
        > `verses.json` post-processed data
    * `raw` folder contains raw data files
* `src` folder contains
    * `config.script` and `config_script_reverse` folder are the folder for default configuration. Because the way hydra package works, they can only be put her
    
    > `dataset.py` functions and class to further process data for modeling<br>

    > `generate.py` script to generate verses or hooks<br>

    > `training.py` script to train our model, either in standard order or reverse order<br>

    > `utils.py` utility file to generate task specified dataset format and appropriate tokenizer<br>
    
## Execution Order

1. **create virtual environment**
    > Please do. There are so many packages. Next step will automatically install all for you.
    ```bash
    python3 -m venv venv
    source ven/bin/activate
    ``` 
2. **install dependencies**<br>
    ```bash
    pip install -r requirements.txt
    ``` 
3. **preprocess data**<br>
    run `data/preprocessing/get_data_copy.ipynb`
    > this file is to preprocess our data and convert it into a json file
4. **train model**<br>
    run `src/training.py`
    > this file is to train our model, either in normal or reverse order
5. **generate verses**<br>
    run `src/generate.py`
    > this file is to generate rap verses
