import os
import json
import platform

import click

import scripts
import scripts.pretrain
from test import dataset


def main() -> None:
    # Test the pretrain model
    scripts.pretrain.pretrain_model()
    
    
if __name__ == "__main__":
    main()
    