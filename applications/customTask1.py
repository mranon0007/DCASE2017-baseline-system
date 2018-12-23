import ptvsd
ptvsd.enable_attach(address = ('10.148.0.2', 3289), redirect_output=True)
ptvsd.wait_for_attach()

import task1

from cStringIO import StringIO
import sys

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

with Capturing() as output:
    task1.main(sys.argv)

print output