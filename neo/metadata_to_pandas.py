import neo
import pandas as pd
import numpy as np
import quantities as pq
from collections import defaultdict


def _add_column(column_name, column, df=None, to_bool=False):
    """
    Internal function.

    Adds a column with name `column_name` to a pandas DataFrame.
    If no dataframe was passed, a new one is created with this one column.
    The return value is the dataframe, however if a dataframe was passed
    it is also edited inplace.

    Parameter:
    ----------
    column_name: str
        This string is used as column header in the dataframe.
    column: list, array, Quantity array
        Column values. The length needs to be the same as the dataframe.
        If the column values have Quantity units, an additional column is
        created with the unit symbols as string.
    df: pandas.DataFrame, None
        Dataframe to which to add the column.
        If None a new dataframe is created.
    to_bool: bool (default: False)
        If True the column values are transformed to bool.
    Returns:
    --------
        df: pandas.DataFrame
            Dataframe with the added column(s).
    """
    if column is None:
        return df
    if to_bool:
        column = [bool(value) for value in column]
    if df is None:
        df = pd.DataFrame(data={column_name:column})
    else:
        df[column_name] = column
    if type(column) == pq.Quantity:
        df['{}.unit'.format(column_name)] = [element.dimensionality.string
                                             for element in column]
    return df

def _object_to_df(obj, attributes=[], use_array_annotations=True,
                  to_bool=False):
    """
    Internal function.

    Can create a pandas.DataFrame from a bottom level neo object
    (SpikeTrain, AnalogSignal, IrregularlySampledSignal, Epoch, Event).

    Parameter:
    ----------
    obj: neo object
        [SpikeTrain, AnalogSignal, IrregularlySampledSignal, Epoch, Event]
    attributes: list of string
        Attributes of the neo object which have the same length
        as elements in the object.
    use_array_annotations: bool (default: True)
        If true, the array_annotations are also represented in the dataframe.
    to_bool: bool or list of bools
        If True the column values of the corresponding attributes
        are transformed to bool. If a list is passed it, it should have the
        same length as attributes.

    Returns:
    --------
    obj_df: pandas.DataFrame
        DataFrame containing all values of the given attributes
        (and array_annotations) for each element in the neo object.
    """
    if type(to_bool) == bool:
        to_bool = [to_bool]*len(attributes)
    if use_array_annotations:
        obj_df = pd.DataFrame(data=obj.array_annotations)
    else:
        obj_df = None
    for attr, tb in zip(attributes, to_bool):
        if hasattr(obj, attr):
            obj_df = _add_column(df=obj_df, column_name=attr,
                                 column=obj.__getattribute__(attr),
                                 to_bool=tb)
    return obj_df

def _objectlist_to_df(objlist, attributes=[], relations=[],
                      relation_index='name', use_annotations=True):
    """
    Internal function.

    Can create a pandas.DataFrame from a list of bottom level neo objects
    (SpikeTrain, AnalogSignal, IrregularlySampledSignal, Epoch, Event).

    Parameter:
    ----------
    obj: list of neo objects
        [SpikeTrain, AnalogSignal, IrregularlySampledSignal, Epoch, Event]
    attributes: list of string
        Attributes of the neo objects.
    relations: list of string
        Names of linked objects to display.
    relation_index: string (default: 'name')
        Attribute of the linked object to be used to represent the linked
        object in the dataframe.
    use_annotations: bool (default: True)
        If true, the annotations are also represented in the dataframe.

    Returns:
    --------
    obj_df: pandas.DataFrame
        DataFrame containing all values of the given attributes
        (and annotations) for each element in the neo object.
    """
    if use_annotations:
        keys = np.unique([key for obj in objlist
                          for key in obj.annotations.keys()])
        list_annotations = defaultdict(list)
        for obj in objlist:
            for key in keys:
                try:
                    list_annotations[key].append(obj.annotations[key])
                except KeyError:
                    list_annotations[key].append(None)
        objlist_df = pd.DataFrame(data=list_annotations)
    else:
        objlist_df = None
    for attr in attributes:
        objlist_df = _add_column(column_name=attr,
                                 column=[obj.__getattribute__(attr)
                                         for obj in objlist],
                                 df=objlist_df)
    for rel in relations:
        column = [None if obj.__getattribute__(rel) is None else
                  obj.__getattribute__(rel).__getattribute__(relation_index)
                  for obj in objlist]
        _add_column(column_name='{}.{}'.format(rel, relation_index),
                    column=column,
                    df=objlist_df)
    return objlist_df

# bottom level objects (children of Segment)

def AnalogSignal_to_df(asig):
    asig_df = _object_to_df(asig)
    if asig.channel_index is not None:
        chx_df = ChannelIndex_to_df(asig.channel_index)
        asig_df = asig_df.merge(chx_df, how='outer',
                                left_index=True, right_on='index')
    display(asig)
    return asig_df

def IrregularlySampledSignal_to_df(irrsig):
    return AnalogSignal_to_df(irrsig)

def SpikeTrain_to_df(st):
    attributes = ['left_sweep', 'waveform']
    display(st)
    return _object_to_df(obj=st, attributes=attributes, to_bool=[False, True])

def Epoch_to_df(epoch):
    attributes = ['times', 'durations', 'labels']
    display(epoch)
    return _object_to_df(obj=epoch, attributes=attributes)

def Event_to_df(evt):
    attributes = ['times', 'labels']
    display(evt)
    return _object_to_df(obj=evt, attributes=attributes)

# lists of bottom level objects

def AnalogSignalList_to_df(asigs):
    attributes = ['__class__', 'name', 'description', 'file_origin', 'shape',
                  't_start', 't_stop', 'sampling_rate']
    relations = ['channel_index', 'segment']
    return _objectlist_to_df(objlist=asigs, attributes=attributes,
                             relations=relations)

def SpikeTrainList_to_df(sts):
    attributes = ['__class__', 'name', 'description', 'file_origin', 'shape',
                  't_start', 't_stop', 'sampling_rate']
    relations = ['unit', 'segment']
    return _objectlist_to_df(objlist=sts, attributes=attributes,
                             relations=relations)

def EpochList_to_df(epochs):
    attributes = ['__class__', 'name', 'description', 'file_origin', 'shape']
    relations = ['segment']
    return _objectlist_to_df(objlist=epochs, attributes=attributes,
                             relations=relations)

def EventList_to_df(events):
    return EventList_to_df(events)

# container objects

def Segment_to_df(seg):
    attributes = ['__class__', 'name', 'description', 'file_origin', 'shape']
    relations = ['block']
    display(seg)
    return _objectlist_to_df(objlist=seg.children, attributes=attributes,
                             relations=relations, use_annotations=False)

# def Block_to_df(blk):

def SegmentList_to_df(segs):
    attributes = ['__class__', 'name', 'description', 'file_origin']
    relations = ['block']
    seglist_df =  _objectlist_to_df(objlist=segs, attributes=attributes,
                                    relations=relations)
    children = ['analogsignals', 'spiketrains', 'epochs', 'events']
    for child in children:
        column = [None if obj.__getattribute__(child) is None else
                  obj.__getattribute__(child).__len__()
                  for obj in segs]
        _add_column(column_name='{}.{}'.format(child, '__len__()'),
                    column=column,
                    df=seglist_df)
    return seglist_df

# def BlockList_to_df(blks):

# children of Block

def ChannelIndex_to_df(chx):
    attributes = ['index', 'channel_ids', 'channel_names', 'coordinates']
    return _object_to_df(obj=chx, attributes=attributes,
                         use_array_annotations=False)

# def Units_to_df(units):
