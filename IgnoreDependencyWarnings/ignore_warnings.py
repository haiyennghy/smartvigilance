# Supress unnecessary warnings from Transformers and Matplotlib
import re
import logging
import numpy as np

class IgnoreWarnings():
    def warn(self, *args, **kwargs):
        pass

    def set_global_logging_level(self, level=logging.ERROR, prefices=[""]):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
        for name in logging.root.manager.loggerDict:
            if re.match(prefix_re, name):
                logging.getLogger(name).setLevel(level)