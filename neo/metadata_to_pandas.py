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
    elif not hasattr(column, "__len__"):
        column = [column]
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
                      relation_index='name', use_annotations=True,
                      use_size=False):
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
    init_dict = {}
    if use_annotations:
        init_dict.update(_transpose_dicts(objlist, 'annotations'))
    if use_size:
        init_dict.update(_transpose_dicts(objlist, 'size'))
    if init_dict:
        objlist_df = pd.DataFrame(data=init_dict)
    else:
        objlist_df = None
    for attr in attributes:
        column = [obj.__getattribute__(attr) if hasattr(obj, attr)
                  else None for obj in objlist]
        objlist_df = _add_column(column_name=attr,
                                 column=column,
                                 df=objlist_df)
    for rel in relations:
        column = [obj.__getattribute__(rel).__getattribute__(relation_index)
                  if hasattr(obj, rel) else None for obj in objlist]
        _add_column(column_name='{}.{}'.format(rel, relation_index),
                    column=column,
                    df=objlist_df)
    return objlist_df

def _transpose_dicts(objlist, dict_name):
    keys = np.unique([key for obj in objlist
                      for key in obj.__getattribute__(dict_name).keys()])
    list_annotations = defaultdict(list)
    for obj in objlist:
        for key in keys:
            try:
                list_annotations[key]\
                .append(obj.__getattribute__(dict_name)[key])
            except KeyError:
                list_annotations[key].append(None)
    return list_annotations

# bottom level objects (children of Segment)

def AnalogSignal_to_df(asig, print_overview=True):
    asig_df = _object_to_df(asig)
    if asig.channel_index is not None:
        chx_df = ChannelIndex_to_df(asig.channel_index)
        asig_df = asig_df.merge(chx_df, how='outer',
                                left_index=True, right_on='index')
    if print_overview:
        display(asig)
    return asig_df

def IrregularlySampledSignal_to_df(irrsig, print_overview=True):
    return AnalogSignal_to_df(irrsig, print_overview=print_overview)

def SpikeTrain_to_df(st, print_overview=True):
    attributes = ['left_sweep', 'waveform']
    if print_overview:
        display(st)
    return _object_to_df(obj=st, attributes=attributes, to_bool=[False, True])

def Epoch_to_df(epoch, print_overview=True):
    attributes = ['times', 'durations', 'labels']
    if print_overview:
        display(epoch)
    return _object_to_df(obj=epoch, attributes=attributes)

def Event_to_df(evt, print_overview=True):
    attributes = ['times', 'labels']
    if print_overview:
        display(evt)
    return _object_to_df(obj=evt, attributes=attributes)

# lists of bottom level objects

def AnalogSignalList_to_df(asigs):
    attributes = ['__class__', 'name', 'description', 'file_origin', 'shape',
                  't_start', 't_stop', 'sampling_rate']
    relations = ['channel_index', 'segment']
    return _objectlist_to_df(objlist=asigs, attributes=attributes,
                             relations=relations)

def IrregularlySampledSignalList_to_df(irrsigs):
    attributes = ['__class__', 'name', 'description', 'file_origin', 'shape',
                  't_start', 't_stop']
    relations = ['channel_index', 'segment']
    return _objectlist_to_df(objlist=irrsigs, attributes=attributes,
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

def Segment_to_df(seg, print_overview=True):
    attributes = ['__class__', 'name', 'description', 'file_origin', 'shape',
                  'size']
    if print_overview:
        display(seg)
    return _objectlist_to_df(objlist=seg.children, attributes=attributes,
                             use_annotations=False)

def Block_to_df(blk, print_overview=True):
    return Segment_to_df(blk, print_overview=print_overview)

def ChannelIndex_to_df(chx, print_overview=True):
    attributes = ['index', 'channel_ids', 'channel_names', 'coordinates']
    if print_overview:
        display(chx)
    return _object_to_df(obj=chx, attributes=attributes,
                         use_array_annotations=False)

def Unit_to_df(unit, print_overview=True):
    if print_overview:
        display(unit)
    return SpikeTrainList_to_df(unit.spiketrains)

# lists of container objects

def SegmentList_to_df(segs):
    attributes = ['__class__', 'name', 'description', 'file_origin']
    relations = ['block']
    return _objectlist_to_df(objlist=segs, attributes=attributes,
                             relations=relations, use_size=True)

def BlockList_to_df(blks):
    attributes = ['__class__', 'name', 'description', 'file_origin']
    return  _objectlist_to_df(objlist=blks, attributes=attributes,
                              use_size=True)

def ChannelIndexList_to_df(chxs):
    return SegmentList_to_df(chxs)

def UnitList_to_df(units):
    attributes = ['__class__', 'name', 'description', 'file_origin']
    relations = ['block', 'channel_index']
    return _objectlist_to_df(objlist=units, attributes=attributes,
                             relations=relations, use_size=True)
