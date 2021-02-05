import sys
sys.path.append("../smartvigilance/")

import re


class ValidTokens():

    def __init__(self):
        # frequently mis-tokenized tokens
        self.tokens = {
                    '(b)(4)' : r'\(\s?b\s?\)\s?\(\s?4\s?\)',        # Anonymized
                    '(b)(6)' : r'\(\s?b\s?\)\s?\(\s?6\s?\)',        # Anonymized
                    'mg/dl' : r'\s?mg\s?\/\s?dl\s?',                # Unit
                    'mmol/l' : r'\s?mmol\s?\/\s?l\s?',              # Unit
                    'age1' :  r'\s?[0-9]*\s?-\s?years\s?-\s?old',   # Age/Year
                    'age2' :  r'\s?[0-9]*\s?-\s?year\s?-\s?old',    # Age/Year
                    'age3' :  r'\s?[0-9]*\s?-\s?yrs\s?-\s?old',     # Age/Year
                    'follow-up' : r'\sf\s?\/\s?u',                  # Abbreviation (f/u)
                    'al.' : r'\sal\s?.'                             # Abbreviation (et al.)
                    #1
                    }

        self.units = [
                     'g',       #gram
                     'mmol/l',
                     'mg/dl',
                     'am',      # time
                     'pm',      # time
                     'cm',      # centimeter
                     'm',       # meter
                     'ncm',     # newton centimeter
                     'ncms',    # newton centimeter
                     ]

        # date formats, e.g. 24-7-2018
        self.dates = []

        # check if digit+unit combinations, e.g. 160g, 5mmol/l ...
        def digit_unit_combi(self, token):
            #return True
            pass
