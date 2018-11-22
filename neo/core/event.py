# -*- coding: utf-8 -*-
'''
This module defines :class:`Event`, an array of events.

:class:`Event` derives from :class:`BaseNeo`, from
:module:`neo.core.baseneo`.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import sys
from copy import deepcopy

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo, merge_annotations
from neo.core.epoch import Epoch


PY_VER = sys.version_info[0]


def _new_event(cls, times=None, labels=None, units=None, name=None,
               file_origin=None, description=None,
               annotations=None, segment=None):
    '''
    A function to map Event.__new__ to function that
    does not do the unit checking. This is needed for pickle to work.
    '''
    e = Event(times=times, labels=labels, units=units, name=name, file_origin=file_origin,
              description=description, **annotations)
    e.segment = segment
    return e


class Event(BaseNeo, pq.Quantity):
    '''
    Array of events.

    *Usage*::

        >>> from neo.core import Event
        >>> from quantities import s
        >>> import numpy as np
        >>>
        >>> evt = Event(np.arange(0, 30, 10)*s,
        ...             labels=np.array(['trig0', 'trig1', 'trig2'],
        ...                             dtype='S'))
        >>>
        >>> evt.times
        array([  0.,  10.,  20.]) * s
        >>> evt.labels
        array(['trig0', 'trig1', 'trig2'],
              dtype='|S5')

    *Required attributes/properties*:
        :times: (quantity array 1D) The time of the events.
        :labels: (numpy.array 1D dtype='S') Names or labels for the events.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.

    Note: Any other additional arguments are assumed to be user-specific
    metadata and stored in :attr:`annotations`.

    '''

    _single_parent_objects = ('Segment',)
    _quantity_attr = 'times'
    _necessary_attrs = (('times', pq.Quantity, 1),
                        ('labels', np.ndarray, 1, np.dtype('S')))

    def __new__(cls, times=None, labels=None, units=None, name=None, description=None,
                file_origin=None, **annotations):
        if times is None:
            times = np.array([]) * pq.s
        if labels is None:
            labels = np.array([], dtype='S')
        if units is None:
            # No keyword units, so get from `times`
            try:
                units = times.units
                dim = units.dimensionality
            except AttributeError:
                raise ValueError('you must specify units')
        else:
            if hasattr(units, 'dimensionality'):
                dim = units.dimensionality
            else:
                dim = pq.quantity.validate_dimensionality(units)
        # check to make sure the units are time
        # this approach is much faster than comparing the
        # reference dimensionality
        if (len(dim) != 1 or list(dim.values())[0] != 1 or
                not isinstance(list(dim.keys())[0], pq.UnitTime)):
            ValueError("Unit %s has dimensions %s, not [time]" %
                       (units, dim.simplified))

        obj = pq.Quantity(times, units=dim).view(cls)
        obj.labels = labels
        obj.segment = None
        return obj

    def __init__(self, times=None, labels=None, units=None, name=None, description=None,
                 file_origin=None, **annotations):
        '''
        Initialize a new :class:`Event` instance.
        '''
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)

    def __reduce__(self):
        '''
        Map the __new__ function onto _new_event, so that pickle
        works
        '''
        return _new_event, (self.__class__, np.array(self), self.labels, self.units,
                            self.name, self.file_origin, self.description,
                            self.annotations, self.segment)

    def __array_finalize__(self, obj):
        super(Event, self).__array_finalize__(obj)
        self.labels = getattr(obj, 'labels', None)
        self.annotations = getattr(obj, 'annotations', None)
        self.name = getattr(obj, 'name', None)
        self.file_origin = getattr(obj, 'file_origin', None)
        self.description = getattr(obj, 'description', None)
        self.segment = getattr(obj, 'segment', None)

    def __repr__(self):
        '''
        Returns a string representing the :class:`Event`.
        '''
        # need to convert labels to unicode for python 3 or repr is messed up
        if PY_VER == 3:
            labels = self.labels.astype('U')
        else:
            labels = self.labels
        objs = ['%s@%s' % (label, time) for label, time in zip(labels,
                                                               self.times)]
        return '<Event: %s>' % ', '.join(objs)

    def _repr_pretty_(self, pp, cycle):
        super(Event, self)._repr_pretty_(pp, cycle)

    def rescale(self, units):
        '''
        Return a copy of the :class:`Event` converted to the specified
        units
        '''
        if self.dimensionality == pq.quantity.validate_dimensionality(units):
            return self.copy()
        obj = Event(times=self.times, labels=self.labels, units=units, name=self.name,
                    description=self.description, file_origin=self.file_origin,
                    **self.annotations)
        obj.segment = self.segment
        return obj

    @property
    def times(self):
        return pq.Quantity(self)

    def merge(self, other):
        '''
        Merge the another :class:`Event` into this one.

        The :class:`Event` objects are concatenated horizontally
        (column-wise), :func:`np.hstack`).

        If the attributes of the two :class:`Event` are not
        compatible, and Exception is raised.
        '''
        othertimes = other.times.rescale(self.times.units)
        times = np.hstack([self.times, othertimes]) * self.times.units
        labels = np.hstack([self.labels, other.labels])
        kwargs = {}
        for name in ("name", "description", "file_origin"):
            attr_self = getattr(self, name)
            attr_other = getattr(other, name)
            if attr_self == attr_other:
                kwargs[name] = attr_self
            else:
                kwargs[name] = "merge(%s, %s)" % (attr_self, attr_other)

        merged_annotations = merge_annotations(self.annotations,
                                               other.annotations)
        kwargs.update(merged_annotations)
        return Event(times=times, labels=labels, **kwargs)

    def _copy_data_complement(self, other):
        '''
        Copy the metadata from another :class:`Event`.
        '''
        for attr in ("labels", "name", "file_origin", "description",
                     "annotations"):
            setattr(self, attr, getattr(other, attr, None))

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_ev = cls(times=self.times,
                     labels=self.labels, units=self.units,
                     name=self.name, description=self.description,
                     file_origin=self.file_origin)
        new_ev.__dict__.update(self.__dict__)
        memo[id(self)] = new_ev
        for k, v in self.__dict__.items():
            try:
                setattr(new_ev, k, deepcopy(v, memo))
            except TypeError:
                setattr(new_ev, k, v)
        return new_ev

    def duplicate_with_new_data(self, signal):
        '''
        Create a new :class:`Event` with the same metadata
        but different data
        '''
        new = self.__class__(times=signal)
        new._copy_data_complement(self)
        return new

    def time_slice(self, t_start, t_stop):
        '''
        Creates a new :class:`Event` corresponding to the time slice of
        the original :class:`Event` between (and including) times
        :attr:`t_start` and :attr:`t_stop`. Either parameter can also be None
        to use infinite endpoints for the time interval.
        '''
        _t_start = t_start
        _t_stop = t_stop
        if t_start is None:
            _t_start = -np.inf
        if t_stop is None:
            _t_stop = np.inf

        indices = (self >= _t_start) & (self <= _t_stop)
        new_evt = self[indices]

        return new_evt

    def as_array(self, units=None):
        """
        Return the event times as a plain NumPy array.

        If `units` is specified, first rescale to those units.
        """
        if units:
            return self.rescale(units).magnitude
        else:
            return self.magnitude

    def as_quantity(self):
        """
        Return the event times as a quantities array.
        """
        return self.view(pq.Quantity)

    def to_epoch(self, pairwise=False, durations=None):
        """
        Returns a new Epoch object based on the times and labels in the Event object.

        This method has three modes of action.

        1. By default, an array of `n` event times will be transformed into
           `n-1` epochs, where the end of one epoch is the beginning of the next.
           This assumes that the events are ordered in time; it is the
           responsibility of the caller to check this is the case.
        2. If `pairwise` is True, then the event times will be taken as pairs
           representing the start and end time of an epoch. The number of
           events must be even, otherwise a ValueError is raised.
        3. If `durations` is given, it should be a scalar Quantity or a
           Quantity array of the same size as the Event.
           Each event time is then taken as the start of an epoch of duration
           given by `durations`.

        `pairwise=True` and `durations` are mutually exclusive. A ValueError
        will be raised if both are given.

        If `durations` is given, epoch labels are set to the corresponding
        labels of the events that indicate the epoch start
        If `durations` is not given, then the event labels A and B bounding
        the epoch are used to set the labels of the epochs in the form 'A-B'.
        """

        if pairwise:
            # Mode 2
            if durations is not None:
                raise ValueError("Inconsistent arguments. "
                                 "Cannot give both `pairwise` and `durations`")
            if self.size % 2 != 0:
                raise ValueError("Pairwise conversion of events to epochs"
                                 " requires an even number of events")
            times = self.times[::2]
            durations = self.times[1::2] - times
            labels = np.array(["{}-{}".format(a, b)
                              for a, b in zip(self.labels[::2], self.labels[1::2])])
        elif durations is None:
            # Mode 1
            times = self.times[:-1]
            durations = np.diff(self.times)
            labels = np.array(["{}-{}".format(a, b)
                              for a, b in zip(self.labels[:-1], self.labels[1:])])
        else:
            # Mode 3
            times = self.times
            labels = self.labels
        return Epoch(times=times, durations=durations, labels=labels)
