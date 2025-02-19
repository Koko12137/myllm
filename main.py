import os
import json
import platform

import click

import test
import scripts
import scripts.pretrain
from test import dataset
import test.pretrain_qwen
import test.qwen2_flashattn


def main() -> None:
    # test.pretrain_qwen.pretrain_model()
    scripts.pretrain.pretrain_model()
    
    
if __name__ == "__main__":
    main()
    