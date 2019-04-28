#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DCASE 2017::Detection of rare sound events / Baseline System


from __future__ import print_function, absolute_import
import sys
import os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

import numpy
import argparse
import textwrap
import platform

from dcase_framework.application_core import BinarySoundEventAppCore
from dcase_framework.parameters import ParameterContainer
from dcase_framework.utils import *
from dcase_framework.features import FeatureExtractor, FeatureContainer
from dcase_framework.learners import EventDetector

__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)

from dcase_framework.datasets import SoundEventDataset, TUTRareSoundEvents_2017_DevelopmentSet
from dcase_framework.metadata import MetaDataContainer, MetaDataItem

from keras.layers.core import Reshape
from keras.layers import Layer, Flatten, LSTM, concatenate, Input, Dense, Dropout, Lambda, CuDNNLSTM
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.models import Model
# from keras.callbacks import EarlyStopping
from keras.utils import plot_model

import tensorflow as tf

from scipy import signal
import numpy as np
import librosa 

class CustomFeatureExtractor(FeatureExtractor):
    def __init__(self, *args, **kwargs):
        kwargs['valid_extractors'] = [
            'spectr'
        ]
        kwargs['default_parameters'] = {
            # 'zero_crossing_rate': {
            #     'mono': True,                       # [true, false]
            #     'center': True,
            # },
        }

        self.ones = np.ones((44,))/44

        super(CustomFeatureExtractor, self).__init__(*args, **kwargs)

    def _spectr(self, data, params):

        feature_matrix = []
        for channel in range(0, data.shape[0]):
            
            # handle 2 channels
            audio_data = data[channel, :]

            # resample to fix frame rate
            if params.get('fs') != 44100:
                audio_data = librosa.core.resample(audio_data, params.get('fs'), 44100)
            
            frequencies, times, spectrogram = signal.stft(
                audio_data,
                # fs = 22222,
                nperseg = params.get('win_length_samples'),
                noverlap = params.get('hop_length_samples'),
                # window = "hamming",
                window=signal.hamming(params.get('win_length_samples'), sym=False),
                # mode = 'magnitude',
            )

            spectrogram = np.abs(spectrogram).T[:-1, :]

            # Improve this section
            # Compress/Smoothen/Denoise the Spectrogram
            ones = self.ones
            spectrogram_temp = [ np.convolve(x, ones, mode='valid') for x in spectrogram ]
            # spectrogram_temp = np.asarray(spectrogram_temp)
            f = 21 #avg window size
            spectrogram_temp_2 = [ np.nanmean(np.r_[l, 0 + np.zeros((-len(l) % f,))].reshape(-1, f), axis=-1) for l in spectrogram_temp]     
            spectrogram_temp = np.asarray(spectrogram_temp_2)

            feature_matrix.append(spectrogram_temp)

        return feature_matrix

class EventDetectorCNN(EventDetector):
    """Scene classifier with CNN"""
    def __init__(self, *args, **kwargs):
        super(EventDetectorCNN, self).__init__(*args, **kwargs)
        self.method = 'cnn'

    def create_model(self):
        ##########Creating Model
        # KERAS MODEL
        
        #Inputs
        X1_Shape_In  = (40*40*1)
        X1_Shape     = (40,40,1)
        X1           = Input(shape=(X1_Shape_In,))
        output_shape = 2

        #CNN Params
        conv1_filters     = 32
        conv1_kernel_size = 5
        conv2_filters     = 16
        conv2_kernel_size = 5
        pool_size         = (2,2)

        # pool0_size = (22,1)
        # spec_reshape_func = lambda x : x.reshape(X1_Shape)
        spec_reshaper = Reshape(X1_Shape)(X1)
        # pool0 = AveragePooling2D(pool_size=pool0_size)(spec_reshaper)

        #CNN
        conv1         = Conv2D(conv1_filters, kernel_size=conv1_kernel_size, activation='relu')(spec_reshaper)
        pool1         = MaxPooling2D(pool_size=pool_size)(conv1)
        conv2         = Conv2D(conv2_filters, kernel_size=conv2_kernel_size, activation='relu')(pool1)
        pool2         = MaxPooling2D(pool_size=pool_size)(conv2)
        conv_dropout1 = Dropout(.3)(pool2)
        flat          = Flatten()(conv_dropout1)
        hidden1       = Dense(512, activation='relu')(flat)
        conv_dropout2 = Dropout(.3)(hidden1)

        #merge
        out = conv_dropout2
        # out = concatenate([conv_dropout2,lstm_dropout_3],axis=-1)

        #output
        output1 = Dense(512, activation='relu')(out)
        output  = Dense(output_shape, activation='sigmoid')(output1)

        #construct Model
        model = Model(inputs=X1, outputs=output)
        # model = Model(inputs=[X1,X2], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

        self['model'] = model
        return model

    def learn(self, data, annotations, data_filenames=None, **kwargs):
        """Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data

        Returns
        -------
        self

        """

        training_files       = annotations.keys()  # Collect training files
        activity_matrix_dict = self._get_target_matrix_dict(data, annotations)

        #generate validation files
        validation = False
        if self.learner_params.get_path('validation.enable', False):
            validation_files = self._generate_validation(
                annotations=annotations,
                validation_type=self.learner_params.get_path('validation.setup_source'),
                valid_percentage=self.learner_params.get_path('validation.validation_amount', 0.20),
                seed=self.learner_params.get_path('validation.seed')
            )
            training_files = sorted(list(set(training_files) - set(validation_files)))
        else:
            validation_files = []
            # Process validation data
        
        if validation_files:
            X_validation = self.prepare_data(data=data, files=validation_files)
            Y_validation = self.prepare_activity(activity_matrix_dict=activity_matrix_dict, files=validation_files)

            validation = (X_validation, Y_validation)
            if self.show_extra_debug:
                self.logger.debug('  Validation items \t[{validation:d}]'.format(validation=len(X_validation)))

        X_training_temp      = numpy.vstack([data[x].feat[0] for x in training_files])
        Y_training           = numpy.vstack([activity_matrix_dict[x] for x in training_files])

        NUM_OF_SPLITS = self.feature_aggregator.win_length_frames
        # X_training shape (frames x NUM_OF_SPLITS x feat) where NUM_OF_SPLITS is # of timestamps
        X_training = X_training_temp
        # X_training = X_training_temp.reshape(X_training_temp.shape[0],NUM_OF_SPLITS,X_training_temp.shape[1]/NUM_OF_SPLITS) 
        # X_training = numpy.reshape(numpy.swapaxes(X_training,1,2), (X_training.shape[0], X_training.shape[2], X_training.shape[1], 1))

        self.create_model()
        self['model'].fit(x = X_training, y = Y_training, validation_data=validation, 
            batch_size = self.learner_params.get_path('batch_size', None),
            epochs     = self.learner_params.get_path('epochs', 100))
        return self

    def _frame_probabilities(self, feature_data):
        return self.model.predict(x=feature_data).T
    
    def predict(self, feature_data):
        """Predict frame probabilities for given feature matrix

        Parameters
        ----------
        feature_data : numpy.ndarray
            Feature data

        Returns
        -------
        str
            class label

        """

        if isinstance(feature_data, FeatureContainer):
            # If we have featureContainer as input, get feature_data
            feature_data = feature_data.feat[0]

        # Get frame probabilities
        return self._frame_probabilities(feature_data)

class EventDetectorLSTM(EventDetector):
    """Scene classifier with LSTM"""
    def __init__(self, *args, **kwargs):
        super(EventDetectorLSTM, self).__init__(*args, **kwargs)
        self.method = 'lstm'

    def create_model(self):
        #model input
        X2_Shape_In  = (40,(60+0)) #frames, features
        
        #lstm input
        X2_Shape     = (40, 60) #times, features
        output_shape = 1

        #LSTM Params
        lstm_units = 256

        #LSTM
        X2             = Input(shape=(X2_Shape_In[0]*X2_Shape_In[1],))

        lstm_reshaper  = Reshape(X2_Shape)(X2)
        lstm_1         = LSTM(lstm_units,return_sequences=True)(lstm_reshaper)
        lstm_dropout_1 = Dropout(.3)(lstm_1)
        lstm_2         = LSTM(lstm_units,return_sequences=False)(lstm_dropout_1)
        lstm_dropout_2 = Dropout(.3)(lstm_2)
        ff             = Dense(512, activation='relu')(lstm_dropout_2)
        lstm_dropout_3 = Dropout(.3)(ff)

        #merge
        # out = conv_dropout2
        out = lstm_dropout_3
        # out = concatenate([conv_dropout2,lstm_dropout_3],axis=-1)

        #output
        output1 = Dense(512, activation='relu')(out)
        output  = Dense(output_shape, activation='sigmoid')(output1)

        #construct Model
        # model = Model(inputs=X1, outputs=output)
        model = Model(inputs=X2, outputs=output)
        # model = Model(inputs=[X1,X2], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

        self['model'] = model
        return model

    def learn(self, data, annotations, data_filenames=None, **kwargs):
        """Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data

        Returns
        -------
        self

        """

        training_files       = annotations.keys()  # Collect training files
        activity_matrix_dict = self._get_target_matrix_dict(data, annotations)

        #generate validation files
        validation = False
        if self.learner_params.get_path('validation.enable', False):
            validation_files = self._generate_validation(
                annotations=annotations,
                validation_type=self.learner_params.get_path('validation.setup_source'),
                valid_percentage=self.learner_params.get_path('validation.validation_amount', 0.20),
                seed=self.learner_params.get_path('validation.seed')
            )
            training_files = sorted(list(set(training_files) - set(validation_files)))
        else:
            validation_files = []
            # Process validation data
        
        if validation_files:
            X_validation = self.prepare_data(data=data, files=validation_files)
            Y_validation = self.prepare_activity(activity_matrix_dict=activity_matrix_dict, files=validation_files)

            validation = (X_validation, Y_validation)
            if self.show_extra_debug:
                self.logger.debug('  Validation items \t[{validation:d}]'.format(validation=len(X_validation)))

        X_training_temp      = numpy.vstack([data[x].feat[0] for x in training_files])
        Y_training           = numpy.vstack([activity_matrix_dict[x] for x in training_files])

        NUM_OF_SPLITS = self.feature_aggregator.win_length_frames
        # X_training shape (frames x NUM_OF_SPLITS x feat) where NUM_OF_SPLITS is # of timestamps
        X_training = X_training_temp
        # X_training = X_training_temp.reshape(X_training_temp.shape[0],NUM_OF_SPLITS,X_training_temp.shape[1]/NUM_OF_SPLITS) 
        # X_training = numpy.reshape(numpy.swapaxes(X_training,1,2), (X_training.shape[0], X_training.shape[2], X_training.shape[1], 1))

        self.create_model()
        self['model'].fit(x = X_training, y = Y_training, validation_data=validation, 
            batch_size = self.learner_params.get_path('batch_size', 128),
            epochs     = self.learner_params.get_path('epochs', 100))
        return self

    def _frame_probabilities(self, feature_data):
        return self.model.predict(x=feature_data).T

    def predict(self, feature_data):
        """Predict frame probabilities for given feature matrix

        Parameters
        ----------
        feature_data : numpy.ndarray
            Feature data

        Returns
        -------
        str
            class label

        """

        if isinstance(feature_data, FeatureContainer):
            # If we have featureContainer as input, get feature_data
            feature_data = feature_data.feat[0]

        # Get frame probabilities
        return self._frame_probabilities(feature_data)

class EventDetectorCNNLSTM(EventDetector):
    """Scene classifier with CNNLSTM"""
    def __init__(self, *args, **kwargs):
        super(EventDetectorCNNLSTM, self).__init__(*args, **kwargs)
        self.method = 'cnnlstm'

    def create_model(self):
        #model input
        X_Shape_In  = (40,(60+40)) # frames, features
        X1_Shape_In  = (40,40) # cnn
        X2_Shape_In  = (40,60) # lstm

        #cnn input
        X1_Shape     = (40,40,1)

        #lstm input
        X2_Shape     = (40, 60) #times, features

        #output shape
        output_shape = 1

        #LSTM Params
        lstm_units = 256

        #CNN Params
        conv1_filters     = 32
        conv1_kernel_size = 5
        conv2_filters     = 16
        conv2_kernel_size = 5
        pool_size         = (2,2)

        # #CNN
        # X1           = Input(shape=(X1_Shape_In[0]*X1_Shape_In[1],))
        #LSTM
        X2             = Input(shape=(X2_Shape_In[0]*X2_Shape_In[1],))

        #CNNLSTM
        X             = Input(shape=(X_Shape_In[0]*X_Shape_In[1],))
        cnn_reshaper  = Reshape((40,100,1))(X)
        lstm_reshaper = Reshape((40,100))(X)

        #CNN
        # cnn_reshaper = Reshape(X1_Shape)(X1)

        #add layer to reshape (40,100,1) to (40,60:100,1) #first 60 columns are MFCCS

        conv1         = Conv2D(conv1_filters, kernel_size=conv1_kernel_size, activation='relu')(cnn_reshaper)
        pool1         = MaxPooling2D(pool_size=pool_size)(conv1)
        conv2         = Conv2D(conv2_filters, kernel_size=conv2_kernel_size, activation='relu')(pool1)
        pool2         = MaxPooling2D(pool_size=pool_size)(conv2)
        conv_dropout1 = Dropout(.3)(pool2)
        flat          = Flatten()(conv_dropout1)
        hidden1       = Dense(512, activation='relu')(flat)
        conv_dropout2 = Dropout(.3)(hidden1)

        #LSTM
        # lstm_reshaper  = Reshape(X2_Shape)(X2)

        #add layer to reshape (40,100) to (40,0:60)

        lstm_1         = LSTM(lstm_units,return_sequences=True)(lstm_reshaper)

        lstm_dropout_1 = Dropout(.4)(lstm_1)
        lstm_2         = LSTM(lstm_units,return_sequences=False)(lstm_dropout_1)
        lstm_dropout_2 = Dropout(.4)(lstm_2)
        ff             = Dense(512, activation='relu')(lstm_dropout_2)
        lstm_dropout_3 = Dropout(.3)(ff)

        #merge
        # out = conv_dropout2 #cnn
        # out = lstm_dropout_3 #lstm
        out = concatenate([conv_dropout2,lstm_dropout_3],axis=-1) #cnn-lstm

        #output
        output1 = Dense(512, activation='relu')(out)

        # #classifier
        # output = Dense(output_shape, activation='softmax')(output1)
        #detector
        output  = Dense(output_shape, activation='sigmoid')(output1)

        #construct Model
        # model = Model(inputs=X1, outputs=output) #CNN
        # model = Model(inputs=X2, outputs=output) #LSTM
        model = Model(inputs=X, outputs=output) #CNNLSTM
        # model = Model(inputs=[X1,X2], outputs=output)

        # #classifier
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #detector
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

        self['model'] = model
        return model

    def learn(self, data, annotations, data_filenames=None, **kwargs):
        """Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data

        Returns
        -------
        self

        """

        training_files       = annotations.keys()  # Collect training files
        activity_matrix_dict = self._get_target_matrix_dict(data, annotations)

        #generate validation files
        validation = False
        if self.learner_params.get_path('validation.enable', False):
            validation_files = self._generate_validation(
                annotations=annotations,
                validation_type=self.learner_params.get_path('validation.setup_source'),
                valid_percentage=self.learner_params.get_path('validation.validation_amount', 0.20),
                seed=self.learner_params.get_path('validation.seed')
            )
            training_files = sorted(list(set(training_files) - set(validation_files)))
        else:
            validation_files = []
            # Process validation data
        
        if validation_files:
            X_validation = self.prepare_data(data=data, files=validation_files)
            Y_validation = self.prepare_activity(activity_matrix_dict=activity_matrix_dict, files=validation_files)

            validation = (X_validation, Y_validation)
            if self.show_extra_debug:
                self.logger.debug('  Validation items \t[{validation:d}]'.format(validation=len(X_validation)))

        X_training_temp      = numpy.vstack([data[x].feat[0] for x in training_files])
        Y_training           = numpy.vstack([activity_matrix_dict[x] for x in training_files])

        NUM_OF_SPLITS = self.feature_aggregator.win_length_frames
        # X_training shape (frames x NUM_OF_SPLITS x feat) where NUM_OF_SPLITS is # of timestamps
        X_training = X_training_temp
        # X_training = X_training_temp.reshape(X_training_temp.shape[0],NUM_OF_SPLITS,X_training_temp.shape[1]/NUM_OF_SPLITS) 
        # X_training = numpy.reshape(numpy.swapaxes(X_training,1,2), (X_training.shape[0], X_training.shape[2], X_training.shape[1], 1))

        self.create_model()
        self['model'].fit(x = X_training, y = Y_training, validation_data=validation, 
            batch_size = self.learner_params.get_path('batch_size', 128),
            epochs     = self.learner_params.get_path('epochs', 100))
        return self

    def _frame_probabilities(self, feature_data):
        return self.model.predict(x=feature_data).T

    def predict(self, feature_data):
        """Predict frame probabilities for given feature matrix

        Parameters
        ----------
        feature_data : numpy.ndarray
            Feature data

        Returns
        -------
        str
            class label

        """

        if isinstance(feature_data, FeatureContainer):
            # If we have featureContainer as input, get feature_data
            feature_data = feature_data.feat[0]

        # Get frame probabilities
        return self._frame_probabilities(feature_data)

class Task2AppCore(BinarySoundEventAppCore):
    def __init__(self, *args, **kwargs):
        kwargs['Datasets'] = {
            # 'MyDataset': MyDataset,
        }
        kwargs['Learners'] = {
            'cnn' : EventDetectorCNN,
            'lstm': EventDetectorLSTM,
            'cnnlstm': EventDetectorCNNLSTM
        }
        kwargs['FeatureExtractor'] = CustomFeatureExtractor

        super(Task2AppCore, self).__init__(*args, **kwargs)


def main(argv):
    numpy.random.seed(123456)  # let's make randomization predictable

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            DCASE 2017
            Task 2: Detection of rare sound events
            Baseline System
            ---------------------------------------------
                Tampere University of Technology / Audio Research Group
                Author:  Toni Heittola ( toni.heittola@tut.fi )

            System description


        '''))

    # Setup argument handling
    parser.add_argument('-m', '--mode',
                        choices=('dev', 'challenge'),
                        default=None,
                        help="Selector for system mode",
                        required=False,
                        dest='mode',
                        type=str)

    parser.add_argument('-p', '--parameters',
                        help='parameter file override',
                        dest='parameter_override',
                        required=False,
                        metavar='FILE',
                        type=argument_file_exists)

    parser.add_argument('-s', '--parameter_set',
                        help='Parameter set id',
                        dest='parameter_set',
                        required=False,
                        type=str)

    parser.add_argument('-t', '--testing',
                        # choices=('dev', 'challenge'),
                        default=False,
                        help="testing mode",
                        required=False,
                        dest='testing',
                        type=str)

    parser.add_argument("-n", "--node",
                        help="Node mode",
                        dest="node_mode",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_sets",
                        help="List of available parameter sets",
                        dest="show_set_list",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_datasets",
                        help="List of available datasets",
                        dest="show_dataset_list",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_parameters",
                        help="Show parameters",
                        dest="show_parameters",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_eval",
                        help="Show evaluated setups",
                        dest="show_eval",
                        action='store_true',
                        required=False)

    parser.add_argument("-o", "--overwrite",
                        help="Overwrite mode",
                        dest="overwrite",
                        action='store_true',
                        required=False)

    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)

    # Parse arguments
    args = parser.parse_args()

    if not args.testing:
        # print("im here")
        import ptvsd
        ptvsd.enable_attach(address = ('10.148.0.2', 3289), redirect_output=True)
        ptvsd.wait_for_attach()
        lp = 1

    # Load default parameters from a file
    default_parameters_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'parameters',
                                               os.path.splitext(os.path.basename(__file__))[0]+'.defaults.yaml')
    if args.parameter_set:
        parameters_sets = args.parameter_set.split(',')
    else:
        parameters_sets = [None]

    for parameter_set in parameters_sets:
        # Initialize ParameterContainer
        params = ParameterContainer(
            project_base=os.path.dirname(os.path.realpath(__file__)),
            path_structure={
                'feature_extractor': [
                    'dataset',
                    'feature_extractor.parameters.*'
                ],
                'feature_normalizer': [
                    'dataset',
                    'feature_extractor.parameters.*'
                ],
                'learner': [
                    'dataset',
                    'feature_extractor',
                    'feature_stacker',
                    'feature_normalizer',
                    'feature_aggregator',
                    'learner'
                ],
                'recognizer': [
                    'dataset',
                    'feature_extractor',
                    'feature_stacker',
                    'feature_normalizer',
                    'feature_aggregator',
                    'learner',
                    'recognizer'
                ],
            }
        )

        # Load default parameters from a file
        params.load(filename=default_parameters_filename)

        if args.parameter_override:
            # Override parameters from a file
            params.override(override=args.parameter_override)

        if parameter_set:
            # Override active_set
            params['active_set'] = parameter_set

        # Process parameters
        params.process()

        # Force overwrite
        if args.overwrite:
            params['general']['overwrite'] = True

        # Override dataset mode from arguments
        if args.mode == 'dev':
            # Set dataset to development
            params['dataset']['method'] = 'development'

            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters(section='dataset')

        elif args.mode == 'challenge':
            # Set dataset to training set for challenge
            params['dataset']['method'] = 'challenge_train'
            params['general']['challenge_submission_mode'] = True

            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters(section='dataset')

        if args.node_mode:
            params['general']['log_system_progress'] = False
            params['general']['print_system_progress'] = False

        # Force ascii progress bar under Windows console
        if platform.system() == 'Windows':
            params['general']['use_ascii_progress_bar'] = True

        # Setup logging
        setup_logging(parameter_container=params['logging'])

        app = Task2AppCore(
            name='DCASE 2017::Detection of rare sound events / Baseline System',
            params=params,
            system_desc=params.get('description'),
            system_parameter_set_id=params.get('active_set'),
            setup_label='Development setup',
            log_system_progress=params.get_path('general.log_system_progress'),
            show_progress_in_console=params.get_path('general.print_system_progress'),
            use_ascii_progress_bar=params.get_path('general.use_ascii_progress_bar')
        )

        # Show parameter set list and exit
        if args.show_set_list:
            params_ = ParameterContainer(
                project_base=os.path.dirname(os.path.realpath(__file__))
            ).load(filename=default_parameters_filename)

            if args.parameter_override:
                # Override parameters from a file
                params_.override(override=args.parameter_override)
            if 'sets' in params_:
                app.show_parameter_set_list(set_list=params_['sets'])

            return

        # Show dataset list and exit
        if args.show_dataset_list:
            app.show_dataset_list()
            return

        # Show system parameters
        if params.get_path('general.log_system_parameters') or args.show_parameters:
            app.show_parameters()

        # Show evaluated systems
        if args.show_eval:
            app.show_eval()
            return

        # Initialize application
        # ==================================================
        if params['flow']['initialize'] and not args.testing:
            app.initialize()

        # Extract features for all audio files in the dataset
        # ==================================================
        if params['flow']['extract_features'] or args.testing:
            if args.testing:
                app.feature_extraction(files=["../uploads/" + args.testing])
            else:
                app.feature_extraction()

        # Prepare feature normalizers
        # ==================================================
        if params['flow']['feature_normalizer']:
            app.feature_normalization()

        # System training
        # ==================================================
        if params['flow']['train_system'] and not args.testing:
            app.system_training()

        # System evaluation in development mode
        if not args.mode or args.mode == 'dev' or args.testing:

            # System testing
            # ==================================================
            if params['flow']['test_system'] or args.testing:
                app.system_testing()

            # System evaluation
            # ==================================================
            if params['flow']['evaluate_system']  and not args.testing:
                app.system_evaluation()

        # System evaluation with challenge data
        elif args.mode == 'challenge':
            # Set dataset to testing set for challenge
            params['dataset']['method'] = 'challenge_test'

            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters('dataset')

            if params['general']['challenge_submission_mode']:
                # If in submission mode, save results in separate folder for easier access
                params['path']['recognizer'] = params.get_path('path.recognizer_challenge_output')

            challenge_app = Task2AppCore(
                name='DCASE 2017::Sound Event Detection in Real-life Audio / Baseline System',
                params=params,
                system_desc=params.get('description'),
                system_parameter_set_id=params.get('active_set'),
                setup_label='Evaluation setup',
                log_system_progress=params.get_path('general.log_system_progress'),
                show_progress_in_console=params.get_path('general.print_system_progress'),
                use_ascii_progress_bar=params.get_path('general.use_ascii_progress_bar')
            )

            # Initialize application
            if params['flow']['initialize']:
                challenge_app.initialize()

            # Extract features for all audio files in the dataset
            if params['flow']['extract_features']:
                challenge_app.feature_extraction()

            # System testing
            if params['flow']['test_system']:
                if params['general']['challenge_submission_mode']:
                    params['general']['overwrite'] = True

                challenge_app.system_testing(single_file_per_fold=True)

                if params['general']['challenge_submission_mode']:
                    challenge_app.ui.line(" ")
                    challenge_app.ui.line("Results for the challenge data are stored at ["+params['path']['recognizer_challenge_output']+"]")
                    challenge_app.ui.line(" ")

            # System evaluation if not in challenge submission mode
            if params['flow']['evaluate_system']:
                challenge_app.system_evaluation(single_file_per_fold=True)

    if args.testing:
        print("recognizer:" + params['path']['recognizer'])

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

# python applications/task2Api.py mixture_devtest_babycry_001_f33132b1ff9b5ff4270bc2a289e8287b.wav
# python applications/task2Api.py mixture_devtest_babycry_002_04b75338e2ab48cbd85c666dd859a140.wav
# python applications/task2Api.py mixture_devtest_babycry_003_9d3885eba392ebf059355c864e83908c.wav
# python applications/task2Api.py mixture_devtest_babycry_004_19c4bcb10576524449c83b01fae0b731.wav
# python applications/task2Api.py mixture_devtest_babycry_024_8aac5fc9f11083a35bacb4ee2927c086.wav
# python applications/task2Api.py 
# python applications/task2Api.py 