import task1

import os
import sys
import threading
import time
import datetime

# import ptvsd
# ptvsd.enable_attach(address = ('10.148.0.2', 3289), redirect_output=True)
# ptvsd.wait_for_attach()

dirname = os.path.dirname(__file__)

class OutputGrabber(object):
    """
    Class used to grab standard output or another stream.
    """
    escape_char = "\b"

    def __init__(self, stream=None, threaded=False):
        self.origstream = stream
        self.threaded = threaded
        if self.origstream is None:
            self.origstream = sys.stdout
        self.origstreamfd = self.origstream.fileno()
        self.capturedtext = ""
        # Create a pipe so the stream can be captured:
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        """
        Start capturing the stream data.
        """
        self.capturedtext = ""
        # Save a copy of the stream:
        self.streamfd = os.dup(self.origstreamfd)
        # Replace the original stream with our write pipe:
        os.dup2(self.pipe_in, self.origstreamfd)
        if self.threaded:
            # Start thread that will read the stream:
            self.workerThread = threading.Thread(target=self.readOutput)
            self.workerThread.start()
            # Make sure that the thread is running and os.read() has executed:
            time.sleep(0.01)

    def stop(self):
        """
        Stop capturing the stream data and save the text in `capturedtext`.
        """
        # Print the escape character to make the readOutput method stop:
        self.origstream.write(self.escape_char)
        # Flush the stream to make sure all our data goes in before
        # the escape character:
        self.origstream.flush()
        if self.threaded:
            # wait until the thread finishes so we are sure that
            # we have until the last character:
            self.workerThread.join()
        else:
            self.readOutput()
        # Close the pipe:
        os.close(self.pipe_in)
        os.close(self.pipe_out)
        # Restore the original stream:
        os.dup2(self.streamfd, self.origstreamfd)
        # Close the duplicate stream:
        os.close(self.streamfd)

    def readOutput(self):
        """
        Read the stream data (one byte at a time)
        and save the text in `capturedtext`.
        """
        while True:
            char = os.read(self.pipe_out, 1)
            if not char or self.escape_char in char:
                break
            self.capturedtext += char

# #take an audio file, add it to test list.
# audioFile = sys.argv[1]
# # testFilePath = os.path.join(os.path.join(dirname, "data/test"), audioFile)
# with open(os.path.join(dirname, "data/TUT-acoustic-scenes-2017-development/evaluation_setup/fold1_test.txt"), "w+") as f:
#     f.write("../test/"+audioFile+"\n") # write the new line before

def run(audioFile):
    # get audio path as input
    try:
        if not audioFile: audioFile = sys.argv[1]
    except:
        exit()

    TASK1_PYFILE = "task1.py"
    TASK1_PARAMS = "-o --node --testing " + audioFile
    TESTS_FILE = "test"
    RESULTS_FILE = "results"
    results = ''

    # create uploads folder
    if not (os.path.exists(os.path.join(dirname, 'data', 'uploads'))):
        os.makedirs(os.path.join(dirname, 'data', 'uploads'))

    # create testing file
    testfoldfile_Path = os.path.join(dirname, 'data', 'TUT-acoustic-scenes-2017-evaluation', 'evaluation_setup', TESTS_FILE + ".txt")

    modifiedTime             = os.path.getmtime(testfoldfile_Path)
    timeStamp                = datetime.datetime.fromtimestamp(modifiedTime).strftime("%b-%d-%y-%H:%M:%S")
    testfoldfile_backup_path = testfoldfile_Path+"_"+timeStamp
    os.rename(testfoldfile_Path, testfoldfile_backup_path)


    try:

        testfoldfile      = open(testfoldfile_Path, 'w+')
        testfoldfile.write("../uploads/"+audioFile)
        testfoldfile.close()

        ## Run task 1
        out = OutputGrabber()
        out.start()
        cmnd = 'python '+ os.path.join(dirname, TASK1_PYFILE) + " " +TASK1_PARAMS
        os.system(cmnd)
        out.stop()
        Task1Output = out.capturedtext

        # Get the Results
        Task1Output = Task1Output.splitlines()
        eval_file = Task1Output[0].split(':')[1] + "/"+RESULTS_FILE+".txt"

        with open(eval_file, 'r') as stream:
            stream_lines = stream.readlines()
            print(stream_lines)
            results = dict([ x.strip().split("\t") for x in stream_lines ])

            # Debugging
            for k, v in results.iteritems():
                print k, v
                break

    except:
        pass

    finally:
        os.remove(testfoldfile_Path)
        os.rename(testfoldfile_backup_path, testfoldfile_Path)

# print("+++++++++++++++++++++++++++++=")
# print(Task1Output)

if __name__ == "__main__":
    run(sys.argv[1])

# get audio path as input

# put audio path in fold99
