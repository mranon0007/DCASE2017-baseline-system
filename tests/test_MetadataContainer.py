""" Unit tests for MetadataContainer """

import nose.tools
import sys
import numpy
import os
sys.path.append('..')
from dcase_framework.metadata import MetaDataContainer, FieldValidator
import tempfile

content = [
        {
            'file': 'audio_001.wav',
            'scene_label': 'office',
            'event_label': 'speech',
            'onset': 1.0,
            'offset': 10.0,
        },
        {
            'file': 'audio_001.wav',
            'scene_label': 'office',
            'event_label': 'mouse clicking',
            'onset': 3.0,
            'offset': 5.0,
        },
        {
            'file': 'audio_001.wav',
            'scene_label': 'office',
            'event_label': 'printer',
            'onset': 7.0,
            'offset': 9.0,
        },
        {
            'file': 'audio_002.wav',
            'scene_label': 'meeting',
            'event_label': 'speech',
            'onset': 1.0,
            'offset': 9.0,
        },
        {
            'file': 'audio_002.wav',
            'scene_label': 'meeting',
            'event_label': 'printer',
            'onset': 5.0,
            'offset': 7.0,
        },
    ]

content2 = [
        {
            'file': 'audio_001.wav',
            'scene_label': 'office',
            'event_label': 'speech',
            'onset': 1.0,
            'offset': 1.2,
        },
        {
            'file': 'audio_001.wav',
            'scene_label': 'office',
            'event_label': 'speech',
            'onset': 1.5,
            'offset': 3.0,
        },
        {
            'file': 'audio_001.wav',
            'scene_label': 'office',
            'event_label': 'speech',
            'onset': 4.0,
            'offset': 6.0,
        },
        {
            'file': 'audio_001.wav',
            'scene_label': 'office',
            'event_label': 'speech',
            'onset': 7.0,
            'offset': 8.0,
        },
    ]


def test_formats():
    delimiters = [',', ';', '\t']
    for delimiter in delimiters:
        tmp = tempfile.NamedTemporaryFile('r+', suffix='.txt', dir='/tmp', delete=False)
        try:
            tmp.write('0.5' + delimiter + '0.7\n')
            tmp.write('2.5' + delimiter + '2.7\n')
            tmp.close()

            item = MetaDataContainer().load(filename=tmp.name)[0]
            nose.tools.eq_(item.onset, 0.5)
            nose.tools.eq_(item.offset, 0.7)

        finally:
            os.unlink(tmp.name)

        tmp = tempfile.NamedTemporaryFile('r+', suffix='.txt',  dir='/tmp', delete=False)
        try:
            tmp.write('0.5' + delimiter + '0.7' + delimiter + 'event\n')
            tmp.write('2.5' + delimiter + '2.7' + delimiter + 'event\n')
            tmp.close()

            item = MetaDataContainer().load(filename=tmp.name)[0]
            nose.tools.eq_(item.onset, 0.5)
            nose.tools.eq_(item.offset, 0.7)
            nose.tools.eq_(item.event_label, 'event')

        finally:
            os.unlink(tmp.name)

        tmp = tempfile.NamedTemporaryFile('r+', suffix='.txt', dir='/tmp', delete=False)
        try:
            tmp.write('file.wav' + delimiter + 'scene' + delimiter + '0.5' + delimiter + '0.7' + delimiter + 'event\n')
            tmp.write('file.wav' + delimiter + 'scene' + delimiter + '0.5' + delimiter + '0.7' + delimiter + 'event\n')
            tmp.close()

            item = MetaDataContainer().load(filename=tmp.name)[0]
            nose.tools.eq_(item.onset, 0.5)
            nose.tools.eq_(item.offset, 0.7)
            nose.tools.eq_(item.event_label, 'event')
            nose.tools.eq_(item.file, 'file.wav')
            nose.tools.eq_(item.scene_label, 'scene')

        finally:
            os.unlink(tmp.name)

        tmp = tempfile.NamedTemporaryFile('r+', suffix='.txt', dir='/tmp', delete=False)
        try:
            tmp.write('file.wav' + delimiter + 'scene' + delimiter + '0.5' + delimiter + '0.7' + delimiter + 'event' + delimiter + 'm' + delimiter + 'a1\n')
            tmp.write('file.wav' + delimiter + 'scene' + delimiter + '0.5' + delimiter + '0.7' + delimiter + 'event' + delimiter + 'm' + delimiter + 'a2\n')
            tmp.close()

            item = MetaDataContainer().load(filename=tmp.name)[0]
            nose.tools.eq_(item.onset, 0.5)
            nose.tools.eq_(item.offset, 0.7)
            nose.tools.eq_(item.event_label, 'event')
            nose.tools.eq_(item.file, 'file.wav')
            nose.tools.eq_(item.scene_label, 'scene')
            nose.tools.eq_(item.identifier, 'a1')
            nose.tools.eq_(item.source_label, 'm')

        finally:
            os.unlink(tmp.name)


def test_content():
    meta = MetaDataContainer(content)
    nose.tools.eq_(len(meta), 5)

    # Test content
    nose.tools.eq_(meta[0].file, 'audio_001.wav')
    nose.tools.eq_(meta[0].scene_label, 'office')
    nose.tools.eq_(meta[0].event_label, 'speech')
    nose.tools.eq_(meta[0].onset, 1.0)
    nose.tools.eq_(meta[0].offset, 10.0)

    nose.tools.eq_(meta[4].file, 'audio_002.wav')
    nose.tools.eq_(meta[4].scene_label, 'meeting')
    nose.tools.eq_(meta[4].event_label, 'printer')
    nose.tools.eq_(meta[4].onset, 5.0)
    nose.tools.eq_(meta[4].offset, 7.0)


def test_filter():
    # Test filter by file
    meta = MetaDataContainer(content).filter(filename='audio_002.wav')

    nose.tools.eq_(len(meta), 2)
    nose.tools.eq_(meta[0].file, 'audio_002.wav')
    nose.tools.eq_(meta[0].scene_label, 'meeting')
    nose.tools.eq_(meta[0].event_label, 'speech')
    nose.tools.eq_(meta[0].onset, 1.0)
    nose.tools.eq_(meta[0].offset, 9.0)

    nose.tools.eq_(meta[1].file, 'audio_002.wav')
    nose.tools.eq_(meta[1].scene_label, 'meeting')
    nose.tools.eq_(meta[1].event_label, 'printer')
    nose.tools.eq_(meta[1].onset, 5.0)
    nose.tools.eq_(meta[1].offset, 7.0)

    # Test filter by scene_label
    meta = MetaDataContainer(content).filter(scene_label='office')

    nose.tools.eq_(len(meta), 3)
    nose.tools.eq_(meta[0].file, 'audio_001.wav')
    nose.tools.eq_(meta[0].scene_label, 'office')
    nose.tools.eq_(meta[0].event_label, 'speech')
    nose.tools.eq_(meta[0].onset, 1.0)
    nose.tools.eq_(meta[0].offset, 10.0)

    nose.tools.eq_(meta[1].file, 'audio_001.wav')
    nose.tools.eq_(meta[1].scene_label, 'office')
    nose.tools.eq_(meta[1].event_label, 'mouse clicking')
    nose.tools.eq_(meta[1].onset, 3.0)
    nose.tools.eq_(meta[1].offset, 5.0)

    # Test filter by event_label
    meta = MetaDataContainer(content).filter(event_label='speech')

    nose.tools.eq_(len(meta), 2)
    nose.tools.eq_(meta[0].file, 'audio_001.wav')
    nose.tools.eq_(meta[0].scene_label, 'office')
    nose.tools.eq_(meta[0].event_label, 'speech')
    nose.tools.eq_(meta[0].onset, 1.0)
    nose.tools.eq_(meta[0].offset, 10.0)

    nose.tools.eq_(meta[1].file, 'audio_002.wav')
    nose.tools.eq_(meta[1].scene_label, 'meeting')
    nose.tools.eq_(meta[1].event_label, 'speech')
    nose.tools.eq_(meta[1].onset, 1.0)
    nose.tools.eq_(meta[1].offset, 9.0)


def test_filter_time_segment():
    meta = MetaDataContainer(content).filter_time_segment(onset=5.0)

    nose.tools.eq_(len(meta), 2)
    nose.tools.eq_(meta[0].file, 'audio_001.wav')
    nose.tools.eq_(meta[0].scene_label, 'office')
    nose.tools.eq_(meta[0].event_label, 'printer')
    nose.tools.eq_(meta[0].onset, 7.0)
    nose.tools.eq_(meta[0].offset, 9.0)

    nose.tools.eq_(meta[1].file, 'audio_002.wav')
    nose.tools.eq_(meta[1].scene_label, 'meeting')
    nose.tools.eq_(meta[1].event_label, 'printer')
    nose.tools.eq_(meta[1].onset, 5.0)
    nose.tools.eq_(meta[1].offset, 7.0)

    meta = MetaDataContainer(content).filter_time_segment(onset=5.0, offset=7.0)

    nose.tools.eq_(len(meta), 1)
    nose.tools.eq_(meta[0].file, 'audio_002.wav')
    nose.tools.eq_(meta[0].scene_label, 'meeting')
    nose.tools.eq_(meta[0].event_label, 'printer')
    nose.tools.eq_(meta[0].onset, 5.0)
    nose.tools.eq_(meta[0].offset, 7.0)


def test_process_events():
    meta = MetaDataContainer(content2).process_events(minimum_event_gap=0.5, minimum_event_length=1.0)

    nose.tools.eq_(len(meta), 3)

    nose.tools.eq_(meta[0].file, 'audio_001.wav')
    nose.tools.eq_(meta[0].scene_label, 'office')
    nose.tools.eq_(meta[0].event_label, 'speech')
    nose.tools.eq_(meta[0].onset, 1.5)
    nose.tools.eq_(meta[0].offset, 3.0)

    nose.tools.eq_(meta[1].file, 'audio_001.wav')
    nose.tools.eq_(meta[1].scene_label, 'office')
    nose.tools.eq_(meta[1].event_label, 'speech')
    nose.tools.eq_(meta[1].onset, 4.0)
    nose.tools.eq_(meta[1].offset, 6.0)

    nose.tools.eq_(meta[2].file, 'audio_001.wav')
    nose.tools.eq_(meta[2].scene_label, 'office')
    nose.tools.eq_(meta[2].event_label, 'speech')
    nose.tools.eq_(meta[2].onset, 7.0)
    nose.tools.eq_(meta[2].offset, 8.0)

    meta = MetaDataContainer(content2).process_events(minimum_event_gap=1.0, minimum_event_length=1.0)

    nose.tools.eq_(len(meta), 1)
    nose.tools.eq_(meta[0].file, 'audio_001.wav')
    nose.tools.eq_(meta[0].scene_label, 'office')
    nose.tools.eq_(meta[0].event_label, 'speech')
    nose.tools.eq_(meta[0].onset, 1.5)
    nose.tools.eq_(meta[0].offset, 8.0)


def test_add_time_offset():
    meta = MetaDataContainer(content2).add_time_offset(offset=2.0)

    nose.tools.eq_(len(meta), 4)

    nose.tools.eq_(meta[0].file, 'audio_001.wav')
    nose.tools.eq_(meta[0].scene_label, 'office')
    nose.tools.eq_(meta[0].event_label, 'speech')
    nose.tools.eq_(meta[0].onset, 3.0)
    nose.tools.eq_(meta[0].offset, 3.2)

    nose.tools.eq_(meta[3].file, 'audio_001.wav')
    nose.tools.eq_(meta[3].scene_label, 'office')
    nose.tools.eq_(meta[3].event_label, 'speech')
    nose.tools.eq_(meta[3].onset, 9.0)
    nose.tools.eq_(meta[3].offset, 10.0)


def test_addition():
    meta = MetaDataContainer(content)
    meta2 = MetaDataContainer(content2)

    meta += meta2

    nose.tools.eq_(len(meta), 9)
    nose.tools.eq_(meta[8].file, 'audio_001.wav')
    nose.tools.eq_(meta[8].scene_label, 'office')
    nose.tools.eq_(meta[8].event_label, 'speech')
    nose.tools.eq_(meta[8].onset, 7.0)
    nose.tools.eq_(meta[8].offset, 8.0)


def test_field_validation():
    validator = FieldValidator()

    # is_number
    nose.tools.eq_(validator.is_number('0.1'), True)
    nose.tools.eq_(validator.is_number('-2.1'), True)
    nose.tools.eq_(validator.is_number('123'), True)
    nose.tools.eq_(validator.is_number('-123'), True)
    nose.tools.eq_(validator.is_number('0'), True)

    nose.tools.eq_(validator.is_number('A'), False)
    nose.tools.eq_(validator.is_number('A123'), False)
    nose.tools.eq_(validator.is_number('A 123'), False)
    nose.tools.eq_(validator.is_number('AabbCc'), False)
    nose.tools.eq_(validator.is_number('A.2'), False)

    # is_audiofile
    nose.tools.eq_(validator.is_audiofile('audio.wav'), True)
    nose.tools.eq_(validator.is_audiofile('audio.mp3'), True)
    nose.tools.eq_(validator.is_audiofile('audio.flac'), True)
    nose.tools.eq_(validator.is_audiofile('audio.raw'), True)
    nose.tools.eq_(validator.is_audiofile('path/path/audio.flac'), True)

    nose.tools.eq_(validator.is_audiofile('audio'), False)
    nose.tools.eq_(validator.is_audiofile('123'), False)
    nose.tools.eq_(validator.is_audiofile('54534.232'), False)

    # is_list
    nose.tools.eq_(validator.is_list('test#'), True)
    nose.tools.eq_(validator.is_list('test#test'), True)
    nose.tools.eq_(validator.is_list('test:test'), True)

    nose.tools.eq_(validator.is_list('test'), False)
    nose.tools.eq_(validator.is_list('test-test'), False)
    nose.tools.eq_(validator.is_list('12342.0'), False)

    # is_alpha
    nose.tools.eq_(validator.is_alpha('a', length=1), True)
    nose.tools.eq_(validator.is_alpha('aa', length=2), True)
    nose.tools.eq_(validator.is_alpha('aaa', length=3), True)

    nose.tools.eq_(validator.is_alpha('aaa', length=1), False)
    nose.tools.eq_(validator.is_alpha('aa', length=1), False)
    nose.tools.eq_(validator.is_alpha('aaa', length=2), False)

    nose.tools.eq_(validator.is_alpha('1', length=1), False)


def test_file_list():
    files = MetaDataContainer(content).file_list

    nose.tools.eq_(len(files), 2)
    nose.tools.eq_(files[0], 'audio_001.wav')
    nose.tools.eq_(files[1], 'audio_002.wav')


def test_event_count():
    nose.tools.eq_(MetaDataContainer(content).event_count, len(content))


def test_scene_label_count():
    nose.tools.eq_(MetaDataContainer(content).scene_label_count, 2)


def test_event_label_count():
    nose.tools.eq_(MetaDataContainer(content).event_label_count, 3)


def test_unique_event_labels():
    events = MetaDataContainer(content).unique_event_labels
    nose.tools.eq_(len(events), 3)
    nose.tools.eq_(events[0], 'mouse clicking')
    nose.tools.eq_(events[1], 'printer')
    nose.tools.eq_(events[2], 'speech')


def test_unique_scene_labels():
    scenes = MetaDataContainer(content).unique_scene_labels
    nose.tools.eq_(len(scenes), 2)
    nose.tools.eq_(scenes[0], 'meeting')
    nose.tools.eq_(scenes[1], 'office')


def test_max_event_offset():
    nose.tools.eq_(MetaDataContainer(content).max_offset, 10)



#embed()