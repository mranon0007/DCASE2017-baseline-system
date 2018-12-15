#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DCASE 2017::Acoustic Scene Classification / Baseline System

from __future__ import print_function, absolute_import
import sys
import os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
import numpy
import argparse
import textwrap
import platform

from dcase_framework.application_core import AcousticSceneClassificationAppCore
from dcase_framework.parameters import ParameterContainer
from dcase_framework.utils import *
from dcase_framework.learners import SceneClassifier
from dcase_framework.features import FeatureExtractor
from dcase_framework.datasets import AcousticSceneDataset
from dcase_framework.metadata import MetaDataContainer, MetaDataItem

__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)

def main(argv):
    numpy.random.seed(123456)  # let's make randomization predictable

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "b043_150_160.wav")

    # Example 1, to get feature only without storing them
    feature_repository = FeatureExtractor().extract(audio_file=filename,
                                                    extractor_name='mfcc',
                                                    extractor_params={
                                                        'mfcc': {
                                                            'n_mfcc': 10
                                                        }
                                                    }
                                                    )
    feature_repository['mfcc'].show()

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
