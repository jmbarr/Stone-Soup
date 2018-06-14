# -*- coding: utf-8 -*-
"""Generic readers for Stone Soup.

This is a collection of generic readers for Stone Soup, allowing quick reading
of data that is in common formats.
"""

import csv
from datetime import datetime

import numpy as np
from dateutil.parser import parse

from ..base import Property
from ..types import Detection
from .base import DetectionReader
from .file import TextFileReader


class CSVDetectionReader(DetectionReader, TextFileReader):
    """A simple detection reader for csv files of detections.

    CSV file must have headers, as these are used to determine which fields
    to use to generate the detection.

    Parameters
    ----------
    """

    state_vector_fields = Property(
        [str], doc='List of columns names to be used in state vector')
    time_field = Property(
        str, doc='Name of column to be used as time field')
    time_field_format = Property(
        str, default=None, doc='Optional datetime format')
    timestamp = Property(
        bool, default=False, doc='Treat time field as a timestamp from epoch')

    def detections_gen(self):
        with open(self.path, encoding=self.encoding, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if self.time_field_format is not None:
                    time_field_value = datetime.strptime(
                        row[self.time_field], self.time_field_format)
                elif self.time_field is True:
                    time_field_value = datetime.fromtimestamp(
                        row[self.time_field])
                else:
                    time_field_value = parse(row[self.time_field])

                detect = Detection(np.array(
                    [[row[col_name]] for col_name in self.state_vector_fields],
                    dtype=np.float32), time_field_value)
                yield time_field_value, {detect}