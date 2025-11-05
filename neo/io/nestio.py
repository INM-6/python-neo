"""
Class for reading output files from NEST simulations
( http://www.nest-simulator.org/ ).
Tested with NEST2.10.0

Depends on: numpy, quantities

Supported: Read

Authors: Julia Sprenger, Maximilian Schmidt, Johanna Senk,
Simon Essink, Robin Gutzen, Jasper Albers,
Aitor Morales-Gregorio, Michael Denker

"""

import os.path
import re  # Import re for regex matching
import warnings
from datetime import datetime
import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Block, Segment, SpikeTrain, AnalogSignal
from neo.core.spiketrainlist import SpikeTrainList

value_type_dict = {"V": pq.mV, "I": pq.pA, 
                   "g": pq.CompoundUnit("10^-9*S"), 
                   "no type": pq.dimensionless}


class NestIO(BaseIO):
    """
    Class for reading NEST output files. GDF files for the spike data and DAT
    files for analog signals are possible.

    Usage:
        >>> from neo.io.nestio import NestIO

        >>> files = ['membrane_voltages-1261-0.dat',
                 'spikes-1258-0.gdf']
        >>> r = NestIO(filenames=files)
        >>> seg = r.read_segment(gid_list=[], t_start=400 * pq.ms,
                             t_stop=600 * pq.ms,
                             id_column_gdf=0, time_column_gdf=1,
                             id_column_dat=0, time_column_dat=1,
                             value_columns_dat=2)
    """

    is_readable = True  # class supports reading, but not writing
    is_writable = False

    supported_objects = [SpikeTrain, AnalogSignal, Segment, Block]
    readable_objects = [SpikeTrain, AnalogSignal, Segment, Block]

    has_header = False
    is_streameable = False

    write_params = None  # writing is not supported

    name = 'nest'
    supported_target_objects = ['SpikeTrain', 'AnalogSignal']
    mode = 'file'

    def __init__(self, filenames=None, target_object='SpikeTrain', **kwargs):
        """
        Parameters
        ----------
        filenames: string or list of strings, default=None
            The filename or list of filenames to load.
        target_object : string or list of strings, default='SpikeTrain'
            The type of neo object that should be read out from the input.
            Options are: 'SpikeTrain', 'AnalogSignal'
        kwargs : dict like
            keyword arguments that will be passed to `numpy.loadtxt` see
            https://numpy.org/devdocs/reference/generated/numpy.loadtxt.html
        """
        if target_object not in self.supported_target_objects:
            raise ValueError(f'{target_object} is not a valid object type. '
                             f'Valid values are {self.objects}.')

        # Ensure filenames is always a list
        if isinstance(filenames, str):
            filenames = [filenames]

        # Turn kwargs to attributes
        self.filenames = filenames
        self.target_object = target_object

        self.IOs = [NESTColumnReader(filename, **kwargs) for filename in filenames]

    def __read_analogsignals(
        self, id_list, time_unit, t_start, t_stop,
        sampling_period, id_column, time_column,
        value_columns, value_types, value_units, **args
    ):
        """
        Internal function for reading multiple analog signals at once.
        This function is called by read_analogsignal() and read_segment().

        Arguments
        ----------
        id_list : list of int or None
            The IDs of the senders of time series to load. If None is specified,
            all IDs will be read if the file contains IDs. If the file
            is a NEST 2.x file that only contains times, all senders
            of one file are read into a single AnalogSignal object per file.
        Other parameters: see read_analogsignal().

        Returns
        -------
        analogsignals : list of AnalogSignal
            The requested list of AnalogSignal objects with an annotation 'id'
            corresponding to the sender ID. If the data comes from a NEST 2.x
            file that only contains times, `id` is set to `None`. In addition,
            the AnalogSignals contains an array annotation 'measurement` for
            each time series of the AnalogSignal object that equates to the
            header of the corresponding column for NEST 3.x files, and to the
            column index for NEST 2.x files.
        """

        # checking gid input parameters
        id_list, id_column = self._check_input_ids(id_list, id_column)
        # checking time input parameters
        t_start, t_stop = self._check_input_times(t_start, t_stop, mandatory=False)

        # checking value input parameters
        (value_columns, value_types, value_units) = self._check_input_values_parameters(
            value_columns, value_types, value_units
        )

        # defining standard column order for internal usage
        # [id_column, time_column, value_column1, value_column2, ...]
        column_ids = [id_column, time_column]
        if value_columns is not None:
            column_ids += value_columns
        for i, cid in enumerate(column_ids):
            if cid is None:
                column_ids[i] = -1

        # assert that no single column is assigned twice
        column_list = [id_column, time_column]
        if value_columns is not None:
            column_list += value_columns
        column_list_no_None = [c for c in column_list if c is not None]
        if len(set(column_list_no_None)) < len(column_list_no_None):
            raise ValueError(
                "One or more columns have been specified to contain "
                "the same data. Columns were specified to {column_list_no_None}."
                ""
            )

        # extracting condition and sorting parameters for raw data loading
        (condition, condition_column,
         sorting_column) = self._get_conditions_and_sorting(id_column,
                                                            time_column,
                                                            id_list,
                                                            t_start,
                                                            t_stop)

        analogsignal_list = []
        for col in self.IOs:

            # loading raw data columns
            data = col.get_columns(
                column_indices=column_ids,
                condition=condition,
                condition_column_index=condition_column,
                sorting_column_indices=sorting_column)

            sampling_period = self._check_input_sampling_period(
                                  sampling_period,
                                  time_column,
                                  time_unit,
                                  data)

            # extracting complete gid list for anasig generation
            if not id_list and id_column is not None:
                current_gid_list = np.unique(data[:, id_column])
            else:
                current_gid_list = id_list

            # generate analogsignals for each neuron ID
            for i in current_gid_list:
                selected_ids = self._get_selected_ids(
                    i, id_column, time_column, t_start, t_stop, time_unit,
                    data)

                # extract starting time of analogsignal
                if (time_column is not None) and data.size:
                    anasig_start_time = data[selected_ids[0], 1] * time_unit
                else:
                    # set t_start equal to sampling_period because NEST starts
                    #  recording only after 1 sampling_period
                    anasig_start_time = 1. * sampling_period

                if value_columns is not None:
                    # create one analogsignal per value column requested
                    for v_id, value_column in enumerate(value_columns):
                        signal = data[
                            selected_ids[0]:selected_ids[1], value_column]

                        # create AnalogSignal objects and annotate them with
                        #  the neuron ID
                        analogsignal_list.append(AnalogSignal(
                            signal * value_units[v_id],
                            sampling_period=sampling_period,
                            t_start=anasig_start_time,
                            id=i,
                            file_origin=col.filename,
                            type=value_types[v_id]))
                        # check for correct length of analogsignal
                        assert (analogsignal_list[-1].t_stop
                                == anasig_start_time + len(signal) *
                                sampling_period)
        return analogsignal_list

    def __read_spiketrains(self,
            id_list, time_unit, t_start, t_stop,
            id_column, time_column, **args):
        """
        Internal function for reading multiple spike trains at once.
        This function is called by read_spiketrain() and read_segment().

        Arguments
        ---------
        id_list : list of int or None
            The IDs of the spike trains to load. If None is specified, all
            IDs will be read in the file if the file contains IDs. If the file
            is a NEST 2.x file that only contains times, all spike times
            of one file are read into a single SpikeTrain object per file.
        Other parameters: see read_spiketrain().

        Returns
        -------
        spiketrains : SpikeTrainList
            The requested SpikeTrainList object with an annotation 'id' for each
            SpikeTrain of the list corresponding to the sender ID. If the data
            comes from a NEST 2.x file that only contains times, `id` is set to
            `None`.
        """
        spiketrain_list = SpikeTrainList()

        for col in self.IOs:
            # Resolve id_column and time_column based on header information
            resolved_id_column = id_column
            resolved_time_column = time_column
            resolved_time_offset_column = None

            # Skip NEST 3.x file if it does not contain spike times
            if col.nest3_contains_time_series:
                warnings.warn(
                    f"NEST 3.x file {col.filename} seems to contain a time series. "
                     "Skipping loading file as Neo SpikeTrain object."
                )
                break

            if col.valid_nest3_file:
                # NEST 3.x file with minimum header for spike trains

                # Handle id_column (sender) for NEST 3.x files
                if col.header_indices.get('sender') is not None:
                    if id_column is not None:
                        warnings.warn(
                            f"id_column={id_column} provided, but 'sender' column found in header at index "
                            f"{col.header_indices['sender']}. Using header information."
                        )
                    resolved_id_column = col.header_indices['sender']
                elif id_column is None:
                    # No recognized sender header, set to default for NEST 2.x
                    resolved_id_column = 0

                # Handle time_column (time_ms or time_steps/time_offset) for NEST 3.x files
                if col.header_indices.get('time_ms') is not None:
                    # time_ms column present
                    if time_column is not None:
                        warnings.warn(
                            f"time_column={time_column} provided, but 'time_ms' column found in header at index "
                            f"{col.header_indices['time_ms']}. Using header information."
                        )
                    resolved_time_column = col.header_indices['time_ms']

                    # Override time_unit to milliseconds
                    if time_unit is not None and time_unit is not pq.ms:
                        warnings.warn(
                            "Ignoring time_unit={time_unit} because 'time_ms' column found in header. "
                        )
                    time_unit = pq.ms
                elif (col.header_indices.get('time_steps') is not None and
                      col.header_indices.get('time_offset') is not None):
                    # time_steps and time_offset columns present
                    if time_column is not None:
                        warnings.warn(
                            f"time_column={time_column} provided, but 'time_steps' and 'time_offset' columns "
                            f"found in header at indices {col.header_indices['time_steps']} and "
                            f"{col.header_indices['time_offset']}. Using header information."
                        )
                    resolved_time_column = col.header_indices['time_steps']
                    resolved_time_offset_column = col.header_indices['time_offset']
                elif time_column is None:
                    # No recognized time header, set to default for NEST 2.x
                    resolved_time_column = 1
            else:
                # NEST 2.x file without header or unrecognized header
                num_available_columns = col.data.shape[1]

                # Make sure user specified columns are valid
                if ((id_column is not None) and (id_column > num_available_columns)):
                    raise ValueError(
                        f"Specified ID column index {id_column} "
                        f"is out of range for file {col.filename}."
                    )

                if ((time_column is not None) and (time_column > num_available_columns)):
                    raise ValueError(
                        f"Specified time column index {time_column} "
                        f"is out of range for file {col.filename}."
                    )

                if num_available_columns==2:
                    if id_column is None:
                        resolved_id_column = 0
                    if time_column is None:
                        resolved_time_column = 1
                elif num_available_columns==1:
                    if time_column is None:
                        resolved_time_column = 0
                else:
                    warnings.warn(
                        f"NEST 2.x or otherwise unrecognized file {col.filename} "
                        f"contains more than 2 columns. "
                        f"Skipping loading file as Neo SpikeTrain object."
                    )


            # Assert that the file contains spike times -- this condition must always be true
            assert resolved_time_column is not None

            # Check validity of IDs being in the resolved ID column
            id_list, resolved_id_column = self._check_input_ids(id_list, resolved_id_column)
            # Check validity of start and stop times
            t_start, t_stop = self._check_input_times(t_start, t_stop, mandatory=True)

            # Assert that no single column is assigned twice
            column_test = [resolved_id_column, resolved_time_column, resolved_time_offset_column]
            column_test = [c for c in column_test if c is not None]
            if len(column_test) != len(set(column_test)):
                raise ValueError("One or more columns have been specified to contain the same data.")

            # defining standard column order for internal usage
            # [id_column, time_column, optional: time_offset]
            column_ids = [resolved_id_column, resolved_time_column]
            if resolved_time_offset_column is not None:
                column_ids.append(resolved_time_offset_column)

            # For NEST 2.x files, the ID column could be missing, and only a time column exists
            for i, cid in enumerate(column_ids):
                if cid is None:
                    column_ids[i] = -1

            (condition, condition_column, sorting_column) = self._get_conditions_and_sorting(
                resolved_id_column, resolved_time_column, id_list, t_start, t_stop
            )

            data = col.get_columns(
                column_indices=column_ids,
                condition=condition,
                condition_column_index=condition_column,
                sorting_column_indices=sorting_column)

            # create a list of SpikeTrains for all neuron IDs in gdf_id_list
            # assign spike times to neuron IDs if id_column is given
            if resolved_id_column is not None:
                if (id_list == []) and resolved_id_column is not None:
                    current_file_ids = np.unique(data[:, 0])
                else:
                    current_file_ids = id_list

                for nid in current_file_ids:
                    selected_ids = self._get_selected_ids(nid, 0,
                                                          1, t_start,
                                                          t_stop, time_unit,
                                                          data)
                    times = data[selected_ids[0]:selected_ids[1], 1]

                    # Handle time_steps and time_offset case
                    if resolved_time_offset_column is not None and data.shape[1] > 2:
                        time_offset = data[selected_ids[0]:selected_ids[1], 2]
                        times = times - time_offset
                    else:
                        times = times

                    spiketrain_list.append(SpikeTrain(times, units=time_unit,
                                                      t_start=t_start,
                                                      t_stop=t_stop,
                                                      id=nid,
                                                      file_origin=col.filename,
                                                      **args))

            # if id_column is not given, all spike times are collected in one
            #  spike train with id=None
            else:
                times = data[:, 1]

                # Handle time_steps and time_offset case
                if resolved_time_offset_column is not None and data.shape[1] > 2:
                    time_offset = data[:, 2]
                    times = times - time_offset
                else:
                    times = times

                spiketrain_list.append(SpikeTrain(times, units=time_unit,
                                                  t_start=t_start,
                                                  t_stop=t_stop,
                                                  id=None,
                                                  file_origin=col.filename,
                                                  **args))
        return spiketrain_list

    def _check_input_ids(self, id_list, id_column):
        """
        Checks gid values and column for consistency. Also makes sure that
        None becomes [None] for consistency.

        Arguments
        ---------
        id_list: list of int or None
            IDs to consider.
        id_column: int,
            Index of the column containing the IDs.

        Returns
        -------
        id_list: list of int or None
            Adjusted IDs
        id_column: int
            Adjusted indices of the column containing the IDs.
        """
        if id_list is None:
            id_list = [id_list]

        if None in id_list and id_column is not None:
            raise ValueError(
                "No sender IDs specified but file contains "
                f"sender IDs in column {str(id_column)}. Specify empty list to "
                "retrieve data for all sender IDs."
                ""
            )

        if id_list != [None] and id_column is None:
            raise ValueError(f"Specified sender IDs to be {id_list}, but no ID column " "specified.")
        return id_list, id_column

    def _check_input_times(self, t_start, t_stop, mandatory=True):
        """
        Checks input times for existence and setting default values if
        necessary.

        t_start: pq.quantity.Quantity, start time of the time range to load.
        t_stop: pq.quantity.Quantity, stop time of the time range to load.
        mandatory: bool, if True times can not be None and an error will be
                raised. if False, time values of None will be replaced by
                -infinity or infinity, respectively. default: True.
        """
        if t_stop is None:
            if mandatory:
                raise ValueError("No t_start specified.")
            else:
                t_stop = np.inf * pq.s
        if t_start is None:
            if mandatory:
                raise ValueError("No t_stop specified.")
            else:
                t_start = -np.inf * pq.s

        for time in (t_start, t_stop):
            if not isinstance(time, pq.quantity.Quantity):
                raise TypeError(f"Time value ({time}) is not a quantity.")
        return t_start, t_stop

    def _check_input_values_parameters(self, value_columns, value_types, value_units):
        """
        Checks value parameters for consistency.

        value_columns: int, column id containing the value to load.
        value_types: list of strings, type of values.
        value_units: list of units of the value columns.

        Returns
        adjusted list of [value_columns, value_types, value_units]
        """
        if value_columns is None:
            warnings.warn('No value column was provided.')
            value_types = None
            value_units = None
            return value_columns, value_types, value_units
        if isinstance(value_columns, int):
            value_columns = [value_columns]
        if value_types is None:
            value_types = ["no type"] * len(value_columns)
        elif isinstance(value_types, str):
            value_types = [value_types]

        # translating value types into units as far as possible
        if value_units is None:
            short_value_types = [vtype.split("_")[0] for vtype in value_types]
            if not all([svt in value_type_dict for svt in short_value_types]):
                raise ValueError(f"Can not interpret value types " f'"{value_types}"')
            value_units = [value_type_dict[svt] for svt in short_value_types]

        # checking for same number of value types, units and columns
        if not (len(value_types) == len(value_units) == len(value_columns)):
            raise ValueError(
                "Length of value types, units and columns does "
                f"not match ({len(value_types)},{len(value_units)},{len(value_columns)})"
            )
        if not all([isinstance(vunit, pq.UnitQuantity) for vunit in value_units]):
            raise ValueError("No value unit or standard value type specified.")

        return value_columns, value_types, value_units

    def _check_input_sampling_period(self, sampling_period, time_column, time_unit, data):
        """
        Checks sampling period, times and time unit for consistency.

        sampling_period: pq.quantity.Quantity, sampling period of data to load.
        time_column: int, column id of times in data to load.
        time_unit: pq.quantity.Quantity, unit of time used in the data to load.
        data: numpy array, the data to be loaded / interpreted.

        Returns
        pq.quantities.Quantity object, the updated sampling period.
        """
        if sampling_period is None:
            if time_column is not None:
                data_sampling = np.unique(
                    np.diff(sorted(np.unique(data[:, time_column]))))
                if len(data_sampling) > 1:
                    raise ValueError(f"Different sampling distances found in " "data set ({data_sampling})")
                else:
                    dt = data_sampling[0]
            else:
                raise ValueError('Can not estimate sampling rate without time '
                                 'column index provided.')
            sampling_period = pq.CompoundUnit(str(dt) + '*'
                                              + time_unit.units.u_symbol)
        elif not isinstance(sampling_period, pq.Quantity):
            raise ValueError("sampling_period is not specified as a unit.")
        return sampling_period

    def _get_conditions_and_sorting(self, id_column, time_column, gid_list, t_start, t_stop):
        """
        Calculates the condition, condition_column and sorting_column based on
        other parameters supplied for loading the data.

        id_column: int, id of the column containing gids.
        time_column: int, id of the column containing times.
        gid_list: list of int, gid to be loaded.
        t_start: pq.quantity.Quantity, start of the time range to be loaded.
        t_stop: pq.quantity.Quantity, stop of the time range to be loaded.

        Returns
        updated [condition, condition_column, sorting_column].
        """
        condition, condition_column = None, None
        sorting_column = []
        curr_id = 0
        if (gid_list != [None]) and (gid_list is not None):
            if gid_list != []:

                def condition(x):
                    return x in gid_list

                condition_column = id_column
            sorting_column.append(curr_id)  # Sorting according to gids first
            curr_id += 1
        if time_column is not None:
            sorting_column.append(curr_id)  # Sorting according to time
            curr_id += 1
        elif t_start != -np.inf and t_stop != np.inf:
            warnings.warn("Ignoring t_start and t_stop parameters, because no " "time column id is provided.")
        if sorting_column == []:
            sorting_column = None
        else:
            sorting_column = sorting_column[::-1]
        return condition, condition_column, sorting_column

    def _get_selected_ids(self, gid, id_column, time_column, t_start, t_stop, time_unit, data):
        """
        Calculates the data range to load depending on the selected gid
        and the provided time range (t_start, t_stop)

        gid: int, gid to be loaded.
        id_column: int, id of the column containing gids.
        time_column: int, id of the column containing times.
        t_start: pq.quantity.Quantity, start of the time range to load.
        t_stop: pq.quantity.Quantity, stop of the time range to load.
        time_unit: pq.quantity.Quantity, time unit of the data to load.
        data: numpy array, data to load.

        Returns
        list of selected gids
        """
        gids = np.array([0, data.shape[0]])
        if id_column is not None:
            gids = np.array([np.searchsorted(data[:, id_column], gid, side='left'),
                             np.searchsorted(data[:, id_column], gid, side='right')])
        gid_data = data[gids[0]:gids[1], :]

        # select only requested time range
        id_shifts = np.array([0, 0])
        if time_column is not None:
            id_shifts[0] = np.searchsorted(gid_data[:, time_column], 
                                           t_start.rescale(time_unit).magnitude,
                                           side="left")
            id_shifts[1] = (
                np.searchsorted(gid_data[:, time_column], 
                                t_stop.rescale(time_unit).magnitude, 
                                side="left") 
                - gid_data.shape[0]
            )

        selected_ids = gids + id_shifts
        return selected_ids

    def read_block(
        self,
        gid_list=None,
        time_unit=pq.ms,
        t_start=None,
        t_stop=None,
        sampling_period=None,
        id_column_dat=0,
        time_column_dat=1,
        value_columns_dat=2,
        id_column_gdf=0,
        time_column_gdf=1,
        value_types=None,
        value_units=None,
        lazy=False,
    ):
        if lazy:
            NotImplementedError("Lazy loading is not implemented for NestIO.")

        seg = self.read_segment(gid_list, time_unit, t_start,
                                t_stop, sampling_period, id_column_dat,
                                time_column_dat, value_columns_dat,
                                id_column_gdf, time_column_gdf, value_types,
                                value_units)
        blk = Block(file_origin=seg.file_origin,
                    file_datetime=seg.file_datetime)
        blk.segments.append(seg)
        return blk

    def read_segment(
        self,
        gid_list=None,
        time_unit=pq.ms,
        t_start=None,
        t_stop=None,
        sampling_period=None,
        id_column_dat=0,
        time_column_dat=1,
        value_columns_dat=2,
        id_column_gdf=0,
        time_column_gdf=1,
        value_types=None,
        value_units=None,
        lazy=False,
    ):
        """
        Reads a Segment which contains SpikeTrain(s) with specified neuron IDs
        from the GDF data.

        Arguments
        ----------
        gid_list : list, default: None
            A list of GDF IDs of which to return SpikeTrain(s). gid_list must
            be specified if the GDF file contains neuron IDs, the default None
            then raises an error. Specify an empty list [] to retrieve the
            spike trains of all neurons.
        time_unit : Quantity (time), optional, default: quantities.ms
            The time unit of recorded time stamps in DAT as well as GDF files.
        t_start : Quantity (time), optional, default: 0 * pq.ms
            Start time of SpikeTrain.
        t_stop : Quantity (time), default: None
            Stop time of SpikeTrain. t_stop must be specified, the default None
            raises an error.
        sampling_period : Quantity (frequency), optional, default: None
            Sampling period of the recorded data.
        id_column_dat : int, optional, default: 0
            Column index of neuron IDs in the DAT file.
        time_column_dat : int, optional, default: 1
            Column index of time stamps in the DAT file.
        value_columns_dat : int, optional, default: 2
            Column index of the analog values recorded in the DAT file.
        id_column_gdf : int, optional, default: 0
            Column index of neuron IDs in the GDF file.
        time_column_gdf : int, optional, default: 1
            Column index of time stamps in the GDF file.
        value_types : str, optional, default: None
            Nest data type of the analog values recorded, eg.'V_m', 'I', 'g_e'
        value_units : Quantity (amplitude), default: None
            The physical unit of the recorded signal values.
        lazy : bool, optional, default: False

        Returns
        -------
        seg : Segment
            The Segment contains one SpikeTrain and one AnalogSignal for
            each ID in gid_list.
        """
        if lazy:
            NotImplementedError("Lazy loading is not implemented for NestIO.")

        if isinstance(gid_list, tuple):
            if gid_list[0] > gid_list[1]:
                raise ValueError("The second entry in gid_list must be "
                                 "greater or equal to the first entry.")
            gid_list = range(gid_list[0], gid_list[1] + 1)

        # __read_xxx() needs a list of IDs
        if gid_list is None:
            gid_list = [None]

        # create an empty Segment
        seg = Segment(file_origin=",".join(self.filenames))
        seg.file_datetime = datetime.fromtimestamp(
                                os.stat(self.filenames[-1]).st_mtime)

        # Load analogsignals and attach to Segment
        if 'AnalogSignal' == self.target_object:
            seg.analogsignals = self.__read_analogsignals(
                gid_list,
                time_unit,
                t_start,
                t_stop,
                sampling_period=sampling_period,
                id_column=id_column_dat,
                time_column=time_column_dat,
                value_columns=value_columns_dat,
                value_types=value_types,
                value_units=value_units)
        if 'SpikeTrain' == self.target_object:
            seg.spiketrains = self.__read_spiketrains(
                gid_list, time_unit, t_start, t_stop, id_column=id_column_gdf, time_column=time_column_gdf
            )

        return seg

    def read_analogsignal(
        self, id=None, time_unit=pq.ms, t_start=None, t_stop=None,
        sampling_period=None, id_column=None, time_column=None,
        value_column=None, value_type=np.float64, value_unit=None,
        lazy=False,  **args
    ):
        """
        Reads an AnalogSignal with specified sender ID from the data.

        Arguments
        ----------
        id : int or None
            The ID of the sender of time series to load. The ID must be
            specified if the file contains sender IDs. If the file is a NEST 2.x
            file that only contains times, all times of one file are read into a
            single AnalogSignal object. In this case, `id` is to be set to
            `None`.
            Default: None
        time_unit : Quantity (time)
            The time unit of recorded time stamps. For NEST 3.x files, if times
            are given by the column headers `time_step` and `time_offset`, the
            time is calculated as (`time_steps` * `time_unit` - `time_offset`).
            If times are given by `time_ms`, the value of `time_unit` is ignored
            and milliseconds are used. For NEST 2.x files, `time_unit` directly
            indicates the unit of values in the file.
            Default: quantities.ms
        t_start : Quantity (time)
            Start time of SpikeTrain. `t_start` must be specified.
            Default: None
        t_stop : Quantity (time)
            Stop time of SpikeTrain. `t_stop` must be specified.
            Default: None
        sampling_period : Quantity (time)
            Sampling period of the signal. Only used for NEST 2.x files without
            a time column.
            Default: None
        id_column : int or None
            Column index of sender IDs. If None, the defaults are used. For
            NEST version 2.x, this is 0 (the first column). For NEST version
            3.x, the column is identified by the column header `sender` in the
            file. In this case, `id_column` is ignored, but if not set to
            `None`, a warning is issued that the value conflicts with the
            header information. If the file contains a header, but the column
            headers do not match the expectancy, the header is ignored, the
            value for `id_column` is used, and a warning is issued indicating
            a non-valid NEST data file.
            Default: None
        time_column : int or None
            Column index of time stamps. If None, the defaults are used. For
            NEST version 2.x, this is 1 (the second column). For NEST
            version 3.x, the column is identified by the column header(s) in
            the file. The relevant header is `time_ms` if `time_in_steps` was
            set to `False` on the NEST spike recorder. Otherwise, the relevant
            headers are `time_steps` and `time_offset` if `time_in_steps` was
            set to `True`. In either of these two cases, `time_column` is
            ignored, but if not set to `None`, a warning is issued if the value
            conflicts with the index inferred from the header information
            (`time_ms` or `time_steps` column). If the file contains a header,
            but the column headers do not match the expectancy, the header is
            ignored, the value for `time_column` is used (in the sense of
            `time_ms`), and a warning is issued indicating a non-valid NEST data
            file.
            Default: None
        value_column : int or string or None
            Column index of signal values. NEST version 2.x, this is an integer
            specifying the index of the column. For NEST version 3.x, the column
            can either bei specified by an integer, or can be is identified by
            a string matching the column header(s) in the file. If None, the
            first column that is neither a time or a sender ID is used.
            Default: None
        value_type : np.dtype
            Default: np.float64
        value_unit : Quantity
            If None is specified, the unit is guessed from the column header in
            NEST 3.x files, and otherwise set to `quantities.dimensionless`.
            Default: None
        lazy : bool
            Lazy loading is currently not implemented for NestIO, and this value
            has no effect.
            Default: False

        Returns
        -------
        analogsignal : AnalogSignal
            The requested AnalogSignal object with an annotation 'id'
            corresponding to the sender ID. If the data comes from a NEST 2.x
            file that only contains times, `id` is set to `None`. In addition,
            the AnalogSignal contains an array annotation 'measurement` for
            each time series of the AnalogSignal object that equates to the
            header of the corresponding column for NEST 3.x files, and to the
            column index for NEST 2.x files.
        """
        if lazy:
            NotImplementedError("Lazy loading is not implemented for NestIO.")

        # __read_spiketrains() needs a list of IDs
        return self.__read_analogsignals(
            [id],
            time_unit,
            t_start,
            t_stop,
            sampling_period=sampling_period,
            id_column=id_column,
            time_column=time_column,
            value_columns=value_column,
            value_types=value_type,
            value_units=value_unit,
        )[0]

    # TODO: There is still a bug in the logic here -- if there are multiple files
    #    Being read from, __read_spiketrains will return one spike train per
    #    file -- this will break the expected behavior here
    def read_spiketrain(
            self, id=None, time_unit=pq.ms, t_start=None, t_stop=None,
            id_column=None, time_column=None, lazy=False, **args
    ):
        """
        Reads a SpikeTrain with specified neuron ID from the data file.

        Arguments
        ----------
        id : int
            The ID of the returned SpikeTrain. The ID must be specified if
            the file contains sender IDs. If the file is a NEST 2.x file that
            only contains times, all spike times of one file are read into a
            single SpikeTrain object. In this case, `id` is to be set to `None`.
            Default: None
        time_unit : Quantity (time)
            The time unit of recorded time stamps. For NEST 3.x files, if times
            are given by the column headers `time_step` and `time_offset`, the
            time is calculated as (`time_steps` * `time_unit` - `time_offset`).
            If times are given by `time_ms`, the value of `time_unit` is ignored
            and milliseconds are used. For NEST 2.x files, `time_unit` directly
            indicates the unit of values in the file.
            Default: quantities.ms
        t_start : Quantity (time)
            Start time of SpikeTrain. `t_start` must be specified.
            Default: None
        t_stop : Quantity (time)
            Stop time of SpikeTrain. `t_stop` must be specified.
            Default: None
        id_column : int or None
            Column index of neuron IDs. If None, the defaults are used. For
            NEST version 2.x, this is 0 (the first column). For NEST version
            3.x, the column is identified by the column header `sender` in the
            file. In this case, `id_column` is ignored, but if not set to
            `None`, a warning is issued that the value conflicts with the
            header information. If the file contains a header, but the column
            headers do not match the expectancy, the header is ignored, the
            value for `id_column` is used, and a warning is issued indicating
            a non-valid NEST data file.
            Default: None
        time_column : int or None
            Column index of time stamps. If None, the defaults are used. For
            NEST version 2.x, this is 1 (the second column). For NEST
            version 3.x, the column is identified by the column header(s) in
            the file. The relevant header is `time_ms` if `time_in_steps` was
            set to `False` on the NEST spike recorder. Otherwise, the relevant
            headers are `time_steps` and `time_offset` if `time_in_steps` was
            set to `True`. In either of these two cases, `time_column` is
            ignored, but if not set to `None`, a warning is issued if the value
            conflicts with the index inferred from the header information
            (`time_ms` or `time_steps` column). If the file contains a header,
            but the column headers do not match the expectancy, the header is
            ignored, the value for `time_column` is used (in the sense of
            `time_ms`), and a warning is issued indicating a non-valid NEST data
            file.
            Default: None
        lazy : bool
            Lazy loading is currently not implemented for NestIO, and this value
            has no effect.
            Default: False

        Returns
        -------
        spiketrain : SpikeTrain
            The requested SpikeTrain object with an annotation 'id'
            corresponding to the sender ID. If the data comes from a NEST 2.x
            file that only contains times, `id` is set to `None`.
        """
        if lazy:
            NotImplementedError("Lazy loading is not implemented for NestIO.")

        if (not isinstance(id, int)) and id is not None:
            raise ValueError("ID has to be of type int or None.")

        return self.__read_spiketrains(
            [id], time_unit, t_start, t_stop,
            id_column, time_column, **args)[0]


class NESTColumnReader:
    """
    Class for reading an NEST ASCII file containing multiple columns of data and possibly a header.

    The file may have multiple header lines, which are interpreted if they conform to NEST standards. Header lines
    are identified as the first lines in the file where the first word in not a digit number.

    Arguments
    ---------
    filename: string
        Path to columnar ASCII file to read

    Keyword arguments
    -----------------
    dtype: np.dtype
        Specifies the data type for the data that is read from file. By default, the IO will inspect the first line of
        data. If it contains a '.' character, the type defaults to np.int64, otherwise it defaults to np.float64.

    Other keyword arguments are passed to `numpy.loadtxt()`
        TODO: Decide if we really want to have an automatic decision here regarding the data type based on a period.
        TODO: Previous versions ignored `dtype` as keyword argument and overwrote it. I think it should be left to the user to supply the dtype.
    """

    def __init__(self, filename, **kwargs):
        self.filename = filename

        # Default values for files without header
        self.nest_version = "2.x"
        self.backend_version = "1"
        self.column_names = []

        # All lines before first data line
        header_lines_raw = []
        # Total count of lines to skip for np.loadtxt
        header_size = 0
        # The first line that starts with a digit
        first_data_line_content = None

        with open(self.filename, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                
                stripped_line = line.strip()

                if stripped_line and stripped_line[0].isdigit():
                    # This is the first data line
                    first_data_line_content = stripped_line
                    break
                else:
                    # This is a header line or empty line
                    header_lines_raw.append(stripped_line)
                    header_size += 1

            if first_data_line_content is None:
                raise IOError("No data lines found in file.")

            # Filter out empty lines for strict header format checking
            valid_parsed_header_lines = [l for l in header_lines_raw if l]
            
            if len(valid_parsed_header_lines) >= 3:
                nest_match = re.match(r'# NEST version: (\d+\.\d+\.\d+)', valid_parsed_header_lines[0])
                backend_match = re.match(r'# RecordingBackendASCII version: (\d+)', valid_parsed_header_lines[1])
                
                # Check for the specific header pattern: two version lines followed by column names
                # Example:
                # # NEST version: 3.6.0
                # # RecordingBackendASCII version: 2
                # sender	time_ms	I_syn_ex	I_syn_in	V_m
                if nest_match and backend_match and not valid_parsed_header_lines[2].startswith('#'):
                    self.nest_version = nest_match.group(1)
                    self.backend_version = backend_match.group(1)
                    self.column_names = valid_parsed_header_lines[2].split()
                else:
                    warnings.warn("Unidentified header format. Using default NEST version '2.x', "
                                  "RecordingBackendASCII version '1', and no column names. Skipping header.")
            elif len(valid_parsed_header_lines) > 0:
                # Some header lines exist, but not enough for specific format
                warnings.warn("Unidentified header format. Using default NEST version '2.x', "
                                  "RecordingBackendASCII version '1', and no column names. Skipping header.")

            # Determine dtype if not specified in kwargs
            if 'dtype' not in kwargs:
                if '.' not in first_data_line_content:
                    kwargs['dtype'] = np.int64
                else:
                    kwargs['dtype'] = np.float64

        # Load data using numpy.loadtxt with the determined number of skipped header lines
        try:
            self.data = np.loadtxt(self.filename, skiprows=header_size, **kwargs)
        except Exception as e:
            raise ValueError(f"Error loading data from file {self.filename}: {e}")

        # Ensure data array is 2-dimensional
        if self.data.ndim == 1:
            self.data = self.data[:, np.newaxis]
        elif self.data.ndim != 2:
            raise ValueError("File could not be parsed correctly. Data is not 2-dimensional.")

        # Create dictionary with column names and their indices
        self.header_indices = {}
        for i, col_name in enumerate(self.column_names):
            self.header_indices[col_name] = i

        # Add entries for the standard headers (even if not present in column_names)
        standard_headers = ['sender', 'time_ms', 'time_steps', 'time_offset']
        for header in standard_headers:
            if header not in self.header_indices:
                self.header_indices[header] = None

        # Check if this is a NEST 3.x file, i.e., the headers
        self.valid_nest3_file = False
        if (self.header_indices['sender'] and
                (self.header_indices['time_ms'] or
                 (self.header_indices['time_steps'] and self.header_indices['time_offset']))):
            self.valid_nest3_file = True

        # Determine if there are any columns besides the standard headers
        # For NEST 3.x, this indicates that the data must be a time series
        self.nest3_contains_time_series = self.valid_nest3_file and (len(self.column_names) > len([h for h in standard_headers if h in self.column_names]))

    def get_columns(self, column_indices="all", condition=None, condition_column_index=None, sorting_column_indices=None):
        """
        Returns data from specific columns of the text file, sorted and filtered by user-defined conditions.

        Arguments
        ---------
        column_indices : int, list of int, string
            IDs of columns to extract, where 0 is the first column. If "all" or an empty list or None is specified,
            all columns are returned.
            Default: "all"
        condition : None, function
            If a function is supplied, it is applied to each row to evaluate if it should be included in the result.
            The function accepts as single argument the column data, i.e., an array with the number of samples (rows)
            in the file. The function needs to return a bool value. If None, all rows are returned.
            Default: None
        condition_column_index : int
            ID of the column on which the condition function is applied to. If None and a condition function is
            specified, an error is raised.
            Default: None
        sorting_column_indices : int or list of int,
            Column IDs to sort output by. List entries have to be ordered by increasing sorting priority! If None, no
            sorting is applied.
            Default: None

        Returns
        -------
        numpy array containing the requested data.
        """

        num_available_columns = self.data.shape[1]

        # If all columns are requested, identify the IDs of all existing columns
        if not column_indices or column_indices == "all":
            column_indices = range(num_available_columns)

        # Simplifies the selection of a single column by accepting an integer as input
        if isinstance(column_indices, (int, float)):
            column_indices = [column_indices]

        # Convert column IDs to numpy array of integers; float IDs are truncated
        column_indices = np.array(column_indices, dtype=np.int32)

        # Test if requested columns exist in the file
        if max(column_indices) > num_available_columns - 1:
            raise ValueError(
                f"Cannot load column ID {max(column_indices)}. File contains "
                f"only {num_available_columns} columns."
            )

        if sorting_column_indices is not None:
            if isinstance(sorting_column_indices, int):
                sorting_column_indices = [sorting_column_indices]

            # Convert sorting column IDs to numpy array of integers
            sorting_column_indices = np.array(sorting_column_indices, dtype=np.int32)

            if max(sorting_column_indices) >= num_available_columns:
                raise ValueError(
                    f"Cannot sort by column ID {max(sorting_column_indices)}. File contains "
                    f"only {num_available_columns} columns."
                )

        # Start with whole dataset selected for return
        selected_data = self.data

        # Apply filter condition to rows
        if condition and (condition_column_index is None):
            raise ValueError(
                "Filter condition is provided, but condition_column is not provided.")
        elif (condition_column_index is not None) and (condition is None):
            warnings.warn(
                "Condition column ID provided, but no condition given. All rows will be returned."
            )
        elif (condition is not None) and (condition_column_index is not None):
            condition_function = np.vectorize(condition)
            mask = condition_function(selected_data[:, condition_column_index]).astype(bool)
            selected_data = selected_data[mask, :]

        # Apply sorting if requested
        if sorting_column_indices is not None:
            # Iterative sorting from lowest to highest priority
            # kind='stable' ensures that when two elements have equal values in the current column,
            # their relative order is preserved so that columns remain intact and prior sorting is preseved.
            for col in sorting_column_indices:
                selected_data = selected_data[np.argsort(selected_data[:, col], kind='stable')]

        # Select only requested columns
        selected_data = selected_data[:, column_indices]

        return selected_data
