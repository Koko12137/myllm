import os
import shutil


def io_operation(config: dict) -> None:
    """This function will check if the output folder exists and is empty. If the folder exists and is not empty, 
    it will raise an error. If the folder exists and allow replace, it will remove the folder. If the folder does 
    not exist, it will create the folder.
    
    Args: 
        config (`dict`): 
            The configuration dictionary. It should contain the following keys: 
            - output: The output path
            - replace: A boolean value that indicates whether to replace the existing folder or not.
            
    Returns:
        None        
    
    Raises:
        ValueError: 
            If the directory exists and is not empty and replace is False.
    """
    output = config['output'] # The output path
    
    # Check if the output folder exists 
    exists = os.path.exists(output)
    if exists:
        # Check if is a directory
        exists = os.path.isdir(output)
    
    if exists:
        # Check if the directory is empty
        empty = not os.listdir(output)
    else:
        # If the directory does not exist, create the directory
        os.makedirs(output)
        print(f"Directory {output} is created.") 
        return
        
    # Check if replace is True
    replace = config['replace']
    
    # If the directory exists and is not empty and replace is False
    if not empty and not replace:
        raise ValueError(f"Directory {output} exists and is not empty.")
    elif empty:
        print(f"Directory {output} is empty.")
        return 
    else:
        # If the directory exists and replace is True, remove the output folder
        shutil.rmtree(output)
        print(f"Directory {output} is removed.")
        
    # Create the output folder
    os.makedirs(output)
    print(f"Directory {output} is created.")
    
    assert os.path.exists(output), f"Directory {output} does not exist."
    