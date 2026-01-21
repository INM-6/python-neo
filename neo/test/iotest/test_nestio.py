"""
Tests of neo.io.nestio
"""
import warnings
import unittest
import os
import tempfile
import quantities as pq
import numpy as np
from neo.io.nestio import NestColumnReader
from neo.io.nestio import NestIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestNestIO_Analogsignals(BaseTestIO, unittest.TestCase):
    ioclass = NestIO
    entities_to_download = ["nest"]
    entities_to_test = []

    def test_read_analogsignal(self):
        """
        Tests reading files in the 2 different formats:
        - with GIDs, with times as floats
        - with GIDs, with time as integer
        """
        filename = self.get_local_path('nest/0gid-1time-2gex-3Vm-1261-0.dat')
        r = NestIO(filenames=filename, target_object='AnalogSignal')
        r.read_analogsignal(id=1, t_stop=1000. * pq.ms,
                            sampling_period=pq.ms, lazy=False,
                            id_column=0, time_column=1,
                            value_column=2, value_type='V_m')
        r.read_segment(id_list=[1], t_stop=1000. * pq.ms,
                       sampling_period=pq.ms, lazy=False, id_column_dat=0,
                       time_column_dat=1, value_columns_dat=2,
                       value_types='V_m')

        filename = self.get_local_path('nest/0gid-1time_in_steps-2Vm-1263-0.dat')
        r = NestIO(filenames=filename, target_object='AnalogSignal')
        r.read_analogsignal(id=1, t_stop=1000. * pq.ms,
                            time_unit=pq.CompoundUnit('0.1*ms'),
                            sampling_period=pq.ms, lazy=False,
                            id_column=0, time_column=1,
                            value_column=2, value_type='V_m')
        r.read_segment(id_list=[1], t_stop=1000. * pq.ms,
                       time_unit=pq.CompoundUnit('0.1*ms'),
                       sampling_period=pq.ms, lazy=False, id_column_dat=0,
                       time_column_dat=1, value_columns_dat=2,
                       value_types='V_m')

        filename = self.get_local_path('nest/0gid-1time-2Vm-1259-0.dat')
        r = NestIO(filenames=filename, target_object='AnalogSignal')
        r.read_analogsignal(id=1, t_stop=1000. * pq.ms,
                            time_unit=pq.CompoundUnit('0.1*ms'),
                            sampling_period=pq.ms, lazy=False,
                            id_column=0, time_column=1,
                            value_column=2, value_type='V_m')
        r.read_segment(id_list=[1], t_stop=1000. * pq.ms,
                       time_unit=pq.CompoundUnit('0.1*ms'),
                       sampling_period=pq.ms, lazy=False, id_column_dat=0,
                       time_column_dat=1, value_columns_dat=2,
                       value_types='V_m')

    def test_id_column_none_multiple_neurons(self):
        """
        Tests if function correctly raises an error if the user tries to read
        from a file which does not contain unit IDs, but data for multiple
        units.
        """
        filename = self.get_local_path("nest/0time-1255-0.gdf")
        r = NestIO(filenames=filename)
        with self.assertRaises(ValueError):
            r.read_spiketrain(t_stop=1000. * pq.ms, lazy=False,
                                sampling_period=pq.ms,
                                id_column=None, time_column=0,
                                value_column=1)
            r.read_segment(t_stop=1000. * pq.ms, lazy=False,
                           sampling_period=pq.ms, id_column_gdf=None,
                           time_column_gdf=0)

    def test_values(self):
        """
        Tests if the function returns the correct values.
        """
        filename = self.get_local_path("nest/0gid-1time-2gex-3Vm-1261-0.dat")
        id_to_test = 1
        r = NestIO(filenames=filename, target_object='AnalogSignal')
        seg = r.read_segment(id_list=[id_to_test],
                             t_stop=1000. * pq.ms,
                             sampling_period=pq.ms, lazy=False,
                             id_column_dat=0, time_column_dat=1,
                             value_columns_dat=2, value_types='V_m')

        dat = np.loadtxt(filename)
        target_data = dat[:, 2][np.where(dat[:, 0] == id_to_test)]
        target_data = target_data[:, None]
        st = seg.analogsignals[0]
        np.testing.assert_array_equal(st.magnitude, target_data)

    def test_read_segment(self):
        """
        Tests if signals are correctly stored in a segment.
        """
        filename = self.get_local_path('nest/0gid-1time-2gex-1262-0.dat')
        r = NestIO(filenames=filename, target_object='AnalogSignal')

        id_list_to_test = range(1, 10)
        seg = r.read_segment(
            id_list=id_list_to_test,
            t_stop=1000.0 * pq.ms,
            sampling_period=pq.ms,
            lazy=False,
            id_column_dat=0,
            time_column_dat=1,
            value_columns_dat=2,
            value_types="V_m",
        )

        self.assertTrue(len(seg.analogsignals) == len(id_list_to_test))

        id_list_to_test = []
        seg = r.read_segment(
            id_list=id_list_to_test,
            t_stop=1000.0 * pq.ms,
            sampling_period=pq.ms,
            lazy=False,
            id_column_dat=0,
            time_column_dat=1,
            value_columns_dat=2,
            value_types="V_m",
        )

        self.assertEqual(len(seg.analogsignals), 50)

    def test_read_block(self):
        """
        Tests if signals are correctly stored in a block.
        """
        filename = self.get_local_path('nest/0gid-1time-2gex-1262-0.dat')
        r = NestIO(filenames=filename, target_object='AnalogSignal')

        id_list_to_test = range(1, 10)
        blk = r.read_block(
            id_list=id_list_to_test,
            t_stop=1000.0 * pq.ms,
            sampling_period=pq.ms,
            lazy=False,
            id_column_dat=0,
            time_column_dat=1,
            value_columns_dat=2,
            value_types="V_m",
        )

        self.assertTrue(len(blk.segments[0].analogsignals) == len(id_list_to_test))

    def test_wrong_input(self):
        """
        Tests two cases of wrong user input, namely
        - User does not specify a value column
        - User does not make any specifications
        - User does not define sampling_period as a unit
        - User specifies a non-default value type without
          specifying a value_unit
        - User specifies t_start < 1.*sampling_period
        """
        filename = self.get_local_path('nest/0gid-1time-2gex-1262-0.dat')
        r = NestIO(filenames=filename, target_object='AnalogSignal')
        with self.assertRaises(ValueError):
            r.read_segment(t_stop=1000.0 * pq.ms, lazy=False, id_column_dat=0, time_column_dat=1)
        with self.assertRaises(ValueError):
            r.read_segment()
        with self.assertRaises(ValueError):
            r.read_segment(id_list=[1], t_stop=1000. * pq.ms,
                           sampling_period=1., lazy=False,
                           id_column_dat=0, time_column_dat=1,
                           value_columns_dat=2, value_types='V_m')

        with self.assertRaises(ValueError):
            r.read_segment(
                id_list=[1],
                t_stop=1000.0 * pq.ms,
                sampling_period=pq.ms,
                lazy=False,
                id_column_dat=0,
                time_column_dat=1,
                value_columns_dat=2,
                value_types="U_mem",
            )

    def test_t_start_t_stop(self):
        """
        Test for correct t_start and t_stop values of AnalogSignalArrays.
        """
        filename = self.get_local_path('nest/0gid-1time-2gex-1262-0.dat')
        r = NestIO(filenames=filename, target_object='AnalogSignal')

        t_start_targ = 450.0 * pq.ms
        t_stop_targ = 480.0 * pq.ms

        seg = r.read_segment(
            id_list=[],
            t_start=t_start_targ,
            t_stop=t_stop_targ,
            lazy=False,
            id_column_dat=0,
            time_column_dat=1,
            value_columns_dat=2,
            value_types="V_m",
        )
        anasigs = seg.analogsignals
        for anasig in anasigs:
            self.assertTrue(anasig.t_start == t_start_targ)
            self.assertTrue(anasig.t_stop == t_stop_targ)

    # TODO: This test is not working yet since there are additional warnings,
    #  and the behavior of the function is not fully clear in terms of if
    #  there should be warning or not, and if so, why only for the time column
    #
    # def test_notimeid(self):
    #     """
    #     Test for warning, when no time column id was provided.
    #     """
    #     filename = self.get_local_path('nest/0gid-1time-2gex-1262-0.dat')
    #     r = NestIO(filenames=filename, target_object='AnalogSignal')
    #
    #     t_start_targ = 450.0 * pq.ms
    #     t_stop_targ = 460.0 * pq.ms
    #     sampling_period = pq.CompoundUnit("5*ms")
    #
    #     with warnings.catch_warnings(record=True) as w:
    #         # Cause all warnings to always be triggered.
    #         warnings.simplefilter("always")
    #         seg = r.read_segment(
    #             gid_list=[],
    #             t_start=t_start_targ,
    #             sampling_period=sampling_period,
    #             t_stop=t_stop_targ,
    #             lazy=False,
    #             id_column_dat=0,
    #             time_column_dat=None,
    #             value_columns_dat=2,
    #             value_types="V_m",
    #         )
    #         # Verify number and content of warning
    #         for ww in w:
    #             print(ww.message)
    #         self.assertEqual(1, len(w))
    #         self.assertIn("no time column id", str(w[0].message))
    #     sts = seg.analogsignals
    #     for st in sts:
    #         self.assertTrue(st.t_start == 1 * 5 * pq.ms)
    #         self.assertTrue(st.t_stop == len(st) * sampling_period + 1 * 5 * pq.ms)

    def test_multiple_value_columns(self):
        """
        Test for simultaneous loading of multiple columns from dat file.
        """
        filename = self.get_local_path('nest/0gid-1time-2Vm-3Iex-4Iin-1264-0.dat')
        r = NestIO(filenames=filename, target_object='AnalogSignal')

        sampling_period = pq.CompoundUnit("5*ms")
        seg = r.read_segment(id_list=[1001], value_columns_dat=[2, 3], sampling_period=sampling_period)
        anasigs = seg.analogsignals
        self.assertEqual(len(anasigs), 2)

    def test_single_gid(self):
        filename = self.get_local_path('nest/N1-0gid-1time-2Vm-1265-0.dat')
        r = NestIO(filenames=filename, target_object='AnalogSignal')
        anasig = r.read_analogsignal(id=1, t_stop=1000. * pq.ms,
                                     time_unit=pq.CompoundUnit('0.1*ms'),
                                     sampling_period=pq.ms, lazy=False,
                                     id_column=0, time_column=1,
                                     value_column=2, value_type='V_m')
        assert anasig.annotations['id'] == 1

    def test_no_gid(self):
        filename = self.get_local_path('nest/N1-0time-1Vm-1266-0.dat')
        r = NestIO(filenames=filename, target_object='AnalogSignal')
        anasig = r.read_analogsignal(id=None, t_stop=1000. * pq.ms,
                                     time_unit=pq.CompoundUnit('0.1*ms'),
                                     sampling_period=pq.ms, lazy=False,
                                     id_column=None, time_column=0,
                                     value_column=1, value_type='V_m')
        self.assertEqual(anasig.annotations['id'], None)
        self.assertEqual(len(anasig), 19)

    def test_no_gid_no_time(self):
        filename = self.get_local_path('nest/N1-0Vm-1267-0.dat')
        r = NestIO(filenames=filename, target_object='AnalogSignal')
        anasig = r.read_analogsignal(sampling_period=pq.ms, lazy=False,
                                     id_column=None, time_column=None,
                                     value_column=0, value_type='V_m')
        self.assertEqual(anasig.annotations['id'], None)
        self.assertEqual(len(anasig), 19)


class TestNestIO_Spiketrains(BaseTestIO, unittest.TestCase):
    ioclass = NestIO
    entities_to_download = ["nest"]
    entities_to_test = []

    def test_read_spiketrain_nest2(self):
        """
        Tests reading files in the 4 different formats:
        - without GIDs, with times as floats
        - without GIDs, with times as integers in time steps
        - with GIDs, with times as floats
        - with GIDs, with times as integers in time steps
        """
        test_configs = [
            {
                "path": "nest/nest2/0time-1255-0.gdf",
                "params": {"id_column": None, "time_column": 0},
                "seg_params": {"id_column_gdf": None, "time_column_gdf": 0}
            },
            {
                "path": "nest/nest2/0time_in_steps-1257-0.gdf",
                "params": {"id_column": None, "time_column": 0, "time_unit": pq.CompoundUnit("0.1*ms")},
                "seg_params": {"id_column_gdf": None, "time_column_gdf": 0, "time_unit": pq.CompoundUnit("0.1*ms")}
            },
            {
                "path": "nest/nest2/0gid-1time-1256-0.gdf",
                "params": {"id": 1, "id_column_gdf": 0, "time_column_gdf": 1},
                "seg_params": {"gid_list": [1], "id_column_gdf": 0, "time_column_gdf": 1}
            },
            {
                "path": "nest/nest2/0gid-1time_in_steps-1258-0.gdf",
                "params": {"id": 1, "id_column": 0, "time_column": 1, "time_unit": pq.CompoundUnit("0.1*ms")},
                "seg_params": {"gid_list": [1], "id_column_gdf": 0, "time_column_gdf": 1, "time_unit": pq.CompoundUnit("0.1*ms")}
            }
        ]

        for config in test_configs:
            filename = self.get_local_path(config["path"])
            try:
                r = NestIO(filenames=filename)
                # Assert successful loading by asserting that the NestIO
                # recruited exactly one NestColumnReader, and that this
                # ColumnReader has data attached.
                self.assertEqual(len(r.IOs), 1)
                self.assertTrue(r.IOs[0].data.size > 0)

                # Test read_spiketrain
                r.read_spiketrain(
                    t_start=400.0 * pq.ms,
                    t_stop=500.0 * pq.ms,
                    lazy=False,
                    **config["params"]
                )

                # Test read_segment
                r.read_segment(
                    t_start=400.0 * pq.ms,
                    t_stop=500.0 * pq.ms,
                    lazy=False,
                    **config["seg_params"]
                )
            except Exception as e:
                self.fail(f"NestIO failed to load Nest 2.x file {filename} with error: {e}")

    def test_read_spiketrain_nest3(self):
        """
        Tests reading Nest 3.x files in the 2 different formats:
        - with GIDs, with times as floats
        - with GIDs, with times as integers in time steps
        """
        test_configs = [
            {
                "path": "nest/nest3/precise_spikes_times-19-0.dat",
                "params": {"id": 1, "id_column": 0, "time_column": 1},
                "seg_params": {"gid_list": [1], "id_column_gdf": 0, "time_column_gdf": 1}
            },
            {
                "path": "nest/nest3/precise_spikes_steps-20-0.dat",
                "params": {"id": 1, "id_column": 0, "time_column": 1},
                "seg_params": {"gid_list": [1], "id_column_gdf": 0, "time_column_gdf": 1}
            }
        ]

        for config in test_configs:
            filename = self.get_local_path(config["path"])
            try:
                r = NestIO(filenames=filename)
                # Assert successful loading by asserting that the NestIO
                # recruited exactly one NestColumnReader, and that this
                # ColumnReader has data attached.
                self.assertEqual(len(r.IOs), 1)
                self.assertTrue(r.IOs[0].data.size > 0)

                # Test read_spiketrain
                r.read_spiketrain(
                    t_start=400.0 * pq.ms,
                    t_stop=500.0 * pq.ms,
                    time_unit=pq.CompoundUnit("0.1*ms"),
                    lazy=False,
                    **config["params"]
                )

                # Test read_segment
                r.read_segment(
                    t_start=400.0 * pq.ms,
                    t_stop=500.0 * pq.ms,
                    time_unit=pq.CompoundUnit("0.1*ms"),
                    lazy=False,
                    **config["seg_params"]
                )
            except Exception as e:
                self.fail(f"NestIO failed to load Nest 3.x file {filename} with error: {e}")

    def test_read_integer(self):
        """
        Tests if spike times are actually stored as integers if they are stored
        in time steps in the file.
        """
        filename = self.get_local_path("nest/nest2/0time_in_steps-1257-0.gdf")
        r = NestIO(filenames=filename)
        st = r.read_spiketrain(
            t_start=400.0 * pq.ms,
            t_stop=500.0 * pq.ms,
            time_unit=pq.CompoundUnit("0.1*ms"),
            lazy=False,
            id_column=None,
            time_column=0,
        )
        self.assertTrue(st.magnitude.dtype == np.int64)
        seg = r.read_segment(
            id_list=[None],
            t_start=400.0 * pq.ms,
            t_stop=500.0 * pq.ms,
            time_unit=pq.CompoundUnit("0.1*ms"),
            lazy=False,
            id_column_gdf=None,
            time_column_gdf=0,
        )
        sts = seg.spiketrains
        self.assertTrue(all([st.magnitude.dtype == np.int64 for st in sts]))

        filename = self.get_local_path("nest/0gid-1time_in_steps-1258-0.gdf")
        r = NestIO(filenames=filename)
        st = r.read_spiketrain(
            id=1,
            t_start=400.0 * pq.ms,
            t_stop=500.0 * pq.ms,
            time_unit=pq.CompoundUnit("0.1*ms"),
            lazy=False,
            id_column=0,
            time_column=1,
        )
        self.assertTrue(st.magnitude.dtype == np.int64)
        seg = r.read_segment(
            id_list=[1],
            t_start=400.0 * pq.ms,
            t_stop=500.0 * pq.ms,
            time_unit=pq.CompoundUnit("0.1*ms"),
            lazy=False,
            id_column_gdf=0,
            time_column_gdf=1,
        )
        sts = seg.spiketrains
        self.assertTrue(all([st.magnitude.dtype == np.int64 for st in sts]))

    def test_read_float(self):
        """
        Tests if spike times are stored as floats if they
        are stored as floats in the file.
        """
        filename = self.get_local_path("nest/nest2/0gid-1time-1256-0.gdf")
        r = NestIO(filenames=filename)
        st = r.read_spiketrain(id=1, t_start=400. * pq.ms,
                               t_stop=500. * pq.ms,
                               lazy=False, id_column=0, time_column=1)
        self.assertTrue(st.magnitude.dtype == np.float64)
        seg = r.read_segment(id_list=[1], t_start=400. * pq.ms,
                             t_stop=500. * pq.ms,
                             lazy=False, id_column_gdf=0, time_column_gdf=1)
        sts = seg.spiketrains
        self.assertTrue(all([s.magnitude.dtype == np.float64 for s in sts]))

    def test_values(self):
        """
        Tests if the routine loads the correct numbers from the file.
        """
        id_to_test = 1
        filename = self.get_local_path("nest/0gid-1time-1256-0.gdf")
        r = NestIO(filenames=filename)
        seg = r.read_segment(
            id_list=[id_to_test],
            t_start=400.0 * pq.ms,
            t_stop=500.0 * pq.ms,
            lazy=False,
            id_column_gdf=0,
            time_column_gdf=1,
        )

        dat = np.loadtxt(filename, dtype=np.float64)
        target_data = dat[:, 1][np.where(dat[:, 0] == id_to_test)]

        st = seg.spiketrains[0]
        np.testing.assert_array_equal(st.magnitude, target_data)

    def test_read_segment(self):
        """
        Tests if spiketrains are correctly stored in a segment.
        """
        filename = self.get_local_path("nest/0gid-1time-1256-0.gdf")
        r = NestIO(filenames=filename)

        id_list_to_test = range(1, 10)
        seg = r.read_segment(
            id_list=id_list_to_test,
            t_start=400.0 * pq.ms,
            t_stop=500.0 * pq.ms,
            lazy=False,
            id_column_gdf=0,
            time_column_gdf=1,
        )
        self.assertTrue(len(seg.spiketrains) == len(id_list_to_test))

        id_list_to_test = []
        seg = r.read_segment(
            id_list=id_list_to_test,
            t_start=400.0 * pq.ms,
            t_stop=500.0 * pq.ms,
            lazy=False,
            id_column_gdf=0,
            time_column_gdf=1,
        )
        self.assertTrue(len(seg.spiketrains) == 50)

    def test_read_segment_accepts_range(self):
        """
        Tests if spiketrains can be retrieved by specifying a range of GDF IDs.
        """
        filename = self.get_local_path("nest/0gid-1time-1256-0.gdf")
        r = NestIO(filenames=filename)

        seg = r.read_segment(
            id_list=(10, 39),
            t_start=400.0 * pq.ms,
            t_stop=500.0 * pq.ms,
            lazy=False,
            id_column_gdf=0,
            time_column_gdf=1,
        )
        self.assertEqual(len(seg.spiketrains), 30)

    def test_read_segment_range_is_reasonable(self):
        """
        Tests if error is thrown correctly, when second entry is smaller than
        the first one of the range.
        """
        filename = self.get_local_path("nest/0gid-1time-1256-0.gdf")
        r = NestIO(filenames=filename)

        seg = r.read_segment(
            id_list=(10, 10),
            t_start=400.0 * pq.ms,
            t_stop=500.0 * pq.ms,
            lazy=False,
            id_column_gdf=0,
            time_column_gdf=1,
        )
        self.assertEqual(len(seg.spiketrains), 1)
        with self.assertRaises(ValueError):
            r.read_segment(
                id_list=(10, 9),
                t_start=400.0 * pq.ms,
                t_stop=500.0 * pq.ms,
                lazy=False,
                id_column_gdf=0,
                time_column_gdf=1,
            )

    def test_read_spiketrain_annotates(self):
        """
        Tests if correct annotation is added when reading a spike train.
        """
        filename = self.get_local_path("nest/0gid-1time-1256-0.gdf")
        r = NestIO(filenames=filename)
        ID = 7
        st = r.read_spiketrain(id=ID, t_start=400.0 * pq.ms, t_stop=500.0 * pq.ms)
        self.assertEqual(ID, st.annotations["id"])

    def test_read_segment_annotates(self):
        """
        Tests if correct annotation is added when reading a segment.
        """
        filename = self.get_local_path("nest/0gid-1time-1256-0.gdf")
        r = NestIO(filenames=filename)
        IDs = (5, 11)
        sts = r.read_segment(id_list=(5, 11), t_start=400.0 * pq.ms, t_stop=500.0 * pq.ms)
        for ID in np.arange(5, 12):
            self.assertEqual(ID, sts.spiketrains[ID - 5].annotations["id"])

    def test_adding_custom_annotation(self):
        """
        Tests if custom annotation is correctly added.
        """
        filename = self.get_local_path("nest/0gid-1time-1256-0.gdf")
        r = NestIO(filenames=filename)
        st = r.read_spiketrain(id=0, t_start=400.0 * pq.ms, t_stop=500.0 * pq.ms, layer="L23", population="I")
        self.assertEqual(0, st.annotations.pop("id"))
        self.assertEqual("L23", st.annotations.pop("layer"))
        self.assertEqual("I", st.annotations.pop("population"))
        self.assertEqual({}, st.annotations)

    def test_wrong_input(self):
        """
        Tests two cases of wrong user input, namely
        - User does not specify neuron IDs although the file contains IDs.
        - User does not make any specifications.
        """
        filename = self.get_local_path("nest/0gid-1time-1256-0.gdf")
        r = NestIO(filenames=filename)
        with self.assertRaises(ValueError):
            r.read_segment(t_start=400.0 * pq.ms, t_stop=500.0 * pq.ms, lazy=False, id_column_gdf=0, time_column_gdf=1)
        with self.assertRaises(ValueError):
            r.read_segment()

    def test_t_start_t_stop(self):
        """
        Tests if the t_start and t_stop arguments are correctly processed.
        """
        filename = self.get_local_path("nest/0gid-1time-1256-0.gdf")
        r = NestIO(filenames=filename)

        t_start_targ = 410.0 * pq.ms
        t_stop_targ = 490.0 * pq.ms

        seg = r.read_segment(
            id_list=[], t_start=t_start_targ, t_stop=t_stop_targ, lazy=False, id_column_gdf=0, time_column_gdf=1
        )
        sts = seg.spiketrains
        self.assertTrue(
            np.max([np.max(st.magnitude) for st in sts if len(st) > 0])
            < t_stop_targ.rescale(sts[0].times.units).magnitude
        )
        self.assertTrue(
            np.min([np.min(st.magnitude) for st in sts if len(st) > 0])
            >= t_start_targ.rescale(sts[0].times.units).magnitude
        )

    def test_t_start_undefined_raises_error(self):
        """
        Tests if undefined t_start, i.e., t_start=None raises error.
        """
        filename = self.get_local_path("nest/0gid-1time-1256-0.gdf")
        r = NestIO(filenames=filename)
        with self.assertRaises(ValueError):
            r.read_spiketrain(id=1, t_stop=500.0 * pq.ms, lazy=False, id_column=0, time_column=1)
        with self.assertRaises(ValueError):
            r.read_segment(id_list=[1, 2, 3], t_stop=500.0 * pq.ms, lazy=False, id_column_gdf=0, time_column_gdf=1)

    def test_t_stop_undefined_raises_error(self):
        """
        Tests if undefined t_stop, i.e., t_stop=None raises error.
        """
        filename = self.get_local_path("nest/0gid-1time-1256-0.gdf")
        r = NestIO(filenames=filename)
        with self.assertRaises(ValueError):
            r.read_spiketrain(id=1, t_start=400.0 * pq.ms, lazy=False, id_column=0, time_column=1)
        with self.assertRaises(ValueError):
            r.read_segment(id_list=[1, 2, 3], t_start=400.0 * pq.ms, lazy=False, id_column_gdf=0, time_column_gdf=1)

    def test_gdf_id_illdefined_raises_error(self):
        """
        Tests if ill-defined gdf_id in read_spiketrain (i.e., None, list, or
        empty list) raises error.
        """
        filename = self.get_local_path("nest/0gid-1time-1256-0.gdf")
        r = NestIO(filenames=filename)
        with self.assertRaises(ValueError):
            r.read_spiketrain(id=[], t_start=400.0 * pq.ms, t_stop=500.0 * pq.ms)
        with self.assertRaises(ValueError):
            r.read_spiketrain(id=[1], t_start=400.0 * pq.ms, t_stop=500.0 * pq.ms)

    def test_read_segment_can_return_empty_spiketrains(self):
        """
        Tests if read_segment makes sure that only non-zero spike trains are
        returned.
        """
        filename = self.get_local_path("nest/0gid-1time-1256-0.gdf")
        r = NestIO(filenames=filename)
        seg = r.read_segment(id_list=[], t_start=400.4 * pq.ms, t_stop=400.5 * pq.ms)
        for st in seg.spiketrains:
            self.assertEqual(st.size, 0)

    def test_read_spiketrain_can_return_empty_spiketrain(self):
        """
        Tests if read_spiketrain returns an empty SpikeTrain if no spikes are in
        time range.
        """
        filename = self.get_local_path("nest/0gid-1time-1256-0.gdf")
        r = NestIO(filenames=filename)
        st = r.read_spiketrain(id=0, t_start=400.0 * pq.ms, t_stop=410.0 * pq.ms)
        self.assertEqual(st.size, 0)


# class TestNestIO_multiple_signal_types(BaseTestIO, unittest.TestCase):
#     ioclass = NestIO
#     entities_to_test = []
#     entities_to_download = [
#         'nest'
#     ]
#
#     def test_read_analogsignal_and_spiketrain(self):
#         """
#         Test if spiketrains and analogsignals can be read simultaneously
#         using read_segment
#         """
#         files = ['nest/0gid-1time-2gex-3Vm-1261-0.dat',
#                  'nest/0gid-1time_in_steps-1258-0.gdf']
#         filenames = [self.get_local_path(e) for e in files]
#
#         r = NestIO(filenames=filenames)
#         seg = r.read_segment(gid_list=[], t_start=400 * pq.ms,
#                              t_stop=600 * pq.ms,
#                              id_column_gdf=0, time_column_gdf=1,
#                              id_column_dat=0, time_column_dat=1,
#                              value_columns_dat=2)
#         self.assertEqual(len(seg.spiketrains), 50)
#         self.assertEqual(len(seg.analogsignals), 50)

# ... existing code ...

class TestNestColumnReader(BaseTestIO, unittest.TestCase):
    ioclass = NestIO
    entities_to_download = ["nest"]
    entities_to_test = []

    def setUp(self):
        BaseTestIO.setUp(self)

        # NEST 2.x file (no header)
        filename = self.get_local_path("nest/nest2/0gid-1time-2Vm-3gex-4gin-1260-0.dat")
        self.testIO_v2_multimeter = NestColumnReader(filename=filename)

        filename = self.get_local_path("nest/nest2/0time-1255-0.gdf")
        self.testIO_v2_spikerecorder = NestColumnReader(filename=filename)

        filename = self.get_local_path("nest/nest2/0time_in_steps-1257-0.gdf")
        self.testIO_v2_spikerecorder_steps = NestColumnReader(filename=filename)

        # NEST 3.x file (with header) containing time series of multimeter
        filename = self.get_local_path("nest/nest3/multimeter_1ms-23-0.dat")
        self.testIO_v3_multimeter = NestColumnReader(filename=filename)

        # NEST 3.x file (with header) containing time series of multimeter
        filename = self.get_local_path("nest/nest3/multimeter_res-24-0.dat")
        self.testIO_v3_multimeter_precise = NestColumnReader(filename=filename)

        # NEST 3.x file (with header) containing time series of multimeter
        filename = self.get_local_path("nest/nest3/voltmeter_times-21-0.dat")
        self.testIO_v3_voltmeter = NestColumnReader(filename=filename)

        # NEST 3.x file (with header) containing only spike times
        filename = self.get_local_path("nest/nest3/aeif_spikes_times-17-0.dat")
        self.testIO_v3_spikerecorder = NestColumnReader(filename=filename)

        # NEST 3.x file (with header) containing only spike times with offset
        filename = self.get_local_path("nest/nest3/precise_spikes_steps-20-0.dat")
        self.testIO_v3_spikerecorder_precise = NestColumnReader(filename=filename)

    def test_header_detection_nest3(self):
        """
        Test that a NEST 3.x file header is correctly parsed,
        column names and indices are mapped, version detection is accurate,
        and has_time_series is properly set.
        """
        # Time series of multimeter
        cr = self.testIO_v3_multimeter
        self.assertTrue(cr.is_valid_nest3_file)
        self.assertTrue(cr.nest_version.count(".") == 2)  # e.g., "3.6.0"
        self.assertEqual(cr.backend_version, "2")
        self.assertIsInstance(cr.column_names, list)
        self.assertIsInstance(cr.header_indices, dict)
        self.assertGreater(len(cr.column_names), 2)  # Expected: sender, time_ms, plus signals
        self.assertIn("sender", cr.header_indices)
        self.assertIn("time_ms", cr.header_indices)
        self.assertTrue(cr.has_time_series)

        # Spike times only
        cr = self.testIO_v3_spikerecorder
        self.assertTrue(cr.is_valid_nest3_file)
        self.assertTrue(cr.nest_version.count(".") == 2)  # e.g., "3.6.0"
        self.assertEqual(cr.backend_version, "2")
        self.assertIsInstance(cr.column_names, list)
        self.assertIsInstance(cr.header_indices, dict)
        self.assertEqual(len(cr.column_names), 2)  # Expected: sender, time_ms, plus signals
        self.assertEqual(len(cr.column_names), cr.data.shape[1])
        self.assertIn("sender", cr.header_indices)
        self.assertIn("time_ms", cr.header_indices)
        self.assertFalse(cr.has_time_series)

        # Spike times with offset parameter
        cr = self.testIO_v3_spikerecorder_precise
        self.assertTrue(cr.is_valid_nest3_file)
        self.assertTrue(cr.nest_version.count(".") == 2)  # e.g., "3.6.0"
        self.assertEqual(cr.backend_version, "2")
        self.assertIsInstance(cr.column_names, list)
        self.assertIsInstance(cr.header_indices, dict)
        self.assertEqual(len(cr.column_names), 3)  # Expected: sender, time_ms, time_offset, plus signals
        self.assertEqual(len(cr.column_names), cr.data.shape[1])
        self.assertIn("sender", cr.header_indices)
        self.assertIn("time_ms", cr.header_indices)
        self.assertIn("time_offset", cr.header_indices)
        self.assertFalse(cr.has_time_series)

    def test_header_detection_nest2(self):
        """
        Test that a NEST 2.x file (no header) falls back to default values.
        """
        cr = self.testIO_v2_multimeter
        self.assertFalse(cr.is_valid_nest3_file)
        self.assertEqual(cr.nest_version, "2.x")
        self.assertEqual(cr.backend_version, "1")
        self.assertEqual(cr.column_names, [])
        self.assertEqual(cr.header_indices, {})
        # has_time_series is by definition False for NEST 2.x files
        self.assertFalse(cr.has_time_series)

    def test_malformed_header_ignored(self):
        """
        Checks that files with headers that do not match NEST 3.x format
        raise warnings and revert to default values.
        """
        # We will simulate the situation by modifying the header of a valid file in a temporary file
        orig_path = self.get_local_path("nest/nest3/multimeter_1ms-23-0.dat")
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tf:
            # Write a bogus header that will not be recognized
            tf.write("# FooBar header none\n# recording ascii X\n# bogus header\n")
            tf.flush()
            # Copy a few lines of data from a real file for proper formating
            with open(orig_path) as realfile:
                lines = realfile.readlines()
                data_lines = [l for l in lines if not l.strip().startswith("#") and l.strip()]
                tf.writelines(data_lines[:10])
            tf.flush()

        # File should not be recognized as NEST 3.x file, but instead follow NEST 2.x logic
        # A warning message should be output to inform users of the potential problem
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cr = NestColumnReader(tf.name)
            self.assertTrue(any("Unidentified" in str(ww.message) for ww in w))
        self.assertFalse(cr.is_valid_nest3_file)
        self.assertEqual(cr.nest_version, "2.x")
        self.assertEqual(cr.backend_version, "1")
        self.assertEqual(cr.column_names, [])
        self.assertEqual(cr.header_indices, {})

        # Clean up temp file
        os.remove(tf.name)

    def test_specific_data_loading(self):
        """
        Tests if the contents of a loaded file is in fact read in correctly,
        i.e., if the `data` attribute and the get_columns() method return correct
        data (the first three lines of text).
        """
        # testIO_v3_multimeter
        # Content:
        # 2       1.000   0.000   0.000   -70.600 0.000
        # 3       1.000   0.000   0.000   -70.600 0.000
        # 4       1.000   0.000   0.000   -70.600 0.000
        cr = self.testIO_v3_multimeter
        expected = np.array([
            [2, 1.0, 0.0, 0.0, -70.6, 0.0],
            [3, 1.0, 0.0, 0.0, -70.6, 0.0],
            [4, 1.0, 0.0, 0.0, -70.6, 0.0]
        ])
        np.testing.assert_array_almost_equal(cr.data[:3], expected)
        np.testing.assert_array_almost_equal(cr.get_columns()[:3], expected)

        # testIO_v3_spikerecorder
        # Content:
        # 2       4.700
        # 3       4.500
        # 4       4.500
        cr = self.testIO_v3_spikerecorder
        expected = np.array([
            [2, 4.7],
            [3, 4.5],
            [4, 4.5]
        ])
        np.testing.assert_array_almost_equal(cr.data[:3], expected)
        np.testing.assert_array_almost_equal(cr.get_columns()[:3], expected)

        # testIO_v2_multimeter
        # Content:
        # 1       405.000 -55.383 6.436   0.483
        # 2       405.000 -55.237 6.125   0.975
        # 3       405.000 -55.169 6.192   0.690
        cr = self.testIO_v2_multimeter
        expected = np.array([
            [1, 405.0, -55.383, 6.436, 0.483],
            [2, 405.0, -55.237, 6.125, 0.975],
            [3, 405.0, -55.169, 6.192, 0.690]
        ])
        np.testing.assert_array_almost_equal(cr.data[:3], expected)
        np.testing.assert_array_almost_equal(cr.get_columns()[:3], expected)

        # testIO_v2_spikerecorder
        # Content:
        # 400.700
        # 400.800
        # 401.600
        cr = self.testIO_v2_spikerecorder
        expected = np.array([
            [400.7],
            [400.8],
            [401.6]
        ])
        np.testing.assert_array_almost_equal(cr.data[:3], expected)
        np.testing.assert_array_almost_equal(cr.get_columns()[:3], expected)

    def test_duplicate_column_header_raises(self):
        """
        If a NEST 3.x file declares duplicate column names in header,
        an exception is raised.
        """
        orig_path = self.get_local_path("nest/nest3/multimeter_1ms-23-0.dat")
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tf:
            with open(orig_path) as realfile:
                lines = realfile.readlines()
                # Use first 2 lines, then a duplicate column header
                tf.write(lines[0])
                tf.write(lines[1])
                bad_header = "sender time_ms sender V_m\n"
                tf.write(bad_header)

                # Copy some of the actual data
                data_lines = [l for l in lines[3:] if l.strip()]
                tf.writelines(data_lines[:5])
            tf.flush()

            # Should raise IOError
            with self.assertRaises(IOError):
                NestColumnReader(tf.name)

        # Clean up temp file
        os.remove(tf.name)

    def test_dtype_autodetection(self):
        """
        Test dtype is auto-detected as int or float if not provided, and if
        overwriting the dtype works.
        """
        # NEST 2.x: times as float ms
        # Should be float64 if data lines have "."
        self.assertEqual(self.testIO_v2_spikerecorder.data.dtype, np.float64)

        # NEST 2.x: times as integer steps
        self.assertEqual(self.testIO_v2_spikerecorder_steps.data.dtype, np.int64)

        # NEST 3.x: times as float ms
        self.assertEqual(self.testIO_v3_spikerecorder.data.dtype, np.float64)

        # NEST 3.x: times as float precise steps
        self.assertEqual(self.testIO_v3_spikerecorder_precise.data.dtype, np.float64)

        # NEST 2.x: overwriting the dtype should work
        filename = self.get_local_path("nest/nest2/0time_in_steps-1257-0.gdf")
        cr = NestColumnReader(filename=filename, dtype=np.float64)
        self.assertEqual(cr.data.dtype, np.float64)

    def test_data_loading_shape(self):
        """
        Data should always be loaded as 2D numpy array.
        """
        # 1D file
        self.assertEqual(len(self.testIO_v2_spikerecorder.data.shape), 2)
        self.assertEqual(self.testIO_v2_spikerecorder.data.shape[1], 1)
        # Multi-column file: .dat with multiple columns
        self.assertEqual(len(self.testIO_v2_multimeter.data.shape), 2)
        self.assertEqual(self.testIO_v2_multimeter.data.shape[1], 5)

    def test_get_columns_errors(self):
        """
        Requesting columns outside range or with missing sorting columns raises ValueError.
        """
        for cr in [self.testIO_v2_multimeter, self.testIO_v3_multimeter]:
            n_columns = cr.data.shape[1]
            with self.assertRaises(ValueError):
                cr.get_columns(column_indices=n_columns)
            with self.assertRaises(ValueError):
                cr.get_columns(column_indices=[0, n_columns])
            with self.assertRaises(ValueError):
                cr.get_columns(sorting_column_indices=n_columns)
            with self.assertRaises(ValueError):
                cr.get_columns(sorting_column_indices=[0, n_columns])

    def test_get_columns_identity(self):
        """
        get_columns() should return same data as .data with no args.
        """
        cr = self.testIO_v2_multimeter
        np.testing.assert_array_equal(cr.get_columns(), cr.data)

        cr = self.testIO_v3_multimeter
        np.testing.assert_array_equal(cr.get_columns(), cr.data)

    def test_no_arguments(self):
        """
        Test if data can be read using the default keyword arguments.
        """
        cr = self.testIO_v2_multimeter
        np.testing.assert_array_equal(cr.get_columns(), cr.data)

    def test_get_columns_basic_output(self):
        """
        Test basic selection of get_columns()
        """
        for cr in [self.testIO_v2_multimeter, self.testIO_v3_multimeter, self.testIO_v3_spikerecorder_precise]:
            # Select second column only
            np.testing.assert_array_equal(cr.get_columns(column_indices=1), cr.data[:, [1]])

            # Select first and second column
            np.testing.assert_array_equal(cr.get_columns(column_indices=[0,1]), cr.data[:, [0, 1]])

            # Select the first and second column in reverse order
            np.testing.assert_array_equal(cr.get_columns(column_indices=[1,0]), cr.data[:, [1, 0]])

    def test_get_columns_warnings_and_errors(self):
        """
        get_columns should warn if condition_column_index given without a condition,
        and error if a condition is given without a condition_column_index.
        """
        cr = self.testIO_v2_multimeter
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cr.get_columns(condition_column_index=0)
            self.assertTrue(any("no condition" in str(ww.message) for ww in w))
        with self.assertRaises(ValueError):
            cr.get_columns(condition=lambda x: True)

    def test_column_names_and_indices_nest3(self):
        """
        For NEST 3.x files, the mapping from column_names to header_indices must be correct and unique.
        """
        cr = self.testIO_v3_multimeter
        # Test unique column names
        self.assertEqual(len(cr.column_names), len(set(cr.column_names)))
        # Test all header_indicies point to a valid column ID
        for name in cr.column_names:
            idx = cr.header_indices[name]
            self.assertTrue(0 <= idx < len(cr.column_names))
        # header_indices must have an entry for every standard header, even if None
        for h in cr.standard_headers:
            self.assertIn(h, cr.header_indices)

    def test_standard_headers_and_has_time_series(self):
        """
        Check that has_time_series is set correctly for a file with >2 columns,
        and that standard_headers are correct.
        """
        for cr in [self.testIO_v3_multimeter, self.testIO_v3_multimeter_precise, self.testIO_v3_voltmeter]:
            # At least one standard header is present (sender, time_ms, etc.)
            self.assertTrue(any(h in cr.column_names for h in cr.standard_headers))
            num_signals = len([name for name in cr.column_names if name not in cr.standard_headers])
            if num_signals > 0:
                self.assertTrue(cr.has_time_series)
            else:
                self.assertFalse(cr.has_time_series)

    def test_no_condition(self):
        """
        Test if a missing condition function leads to a warning
        """
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            self.testIO_v2_multimeter.get_columns(condition_column_index=0)
            # Verify number and content of warning
            assert len(w) == 1
            assert "no condition" in str(w[-1].message)

    def test_no_condition_column(self):
        """
        Test if a missing condition column leads to an error
        """
        with self.assertRaises(ValueError) as context:
            self.testIO_v2_multimeter.get_columns(condition=lambda x: True)

        self.assertTrue("but condition_column is not provided" in str(context.exception))

    def test_correct_condition_selection(self):
        """
        Test if combination of condition function and condition_column works
        properly.
        """
        condition_column = 0

        def condition_function(x):
            return x > 10

        result = self.testIO_v2_multimeter.get_columns(condition=condition_function, condition_column_index=0)
        selected_ids = np.where(condition_function(self.testIO_v2_multimeter.data[:, condition_column]))[0]
        expected = self.testIO_v2_multimeter.data[selected_ids, :]

        np.testing.assert_array_equal(result, expected)

        assert all(condition_function(result[:, condition_column]))

    def test_sorting(self):
        """
        Test if the sorting of columns works properly.
        """

        # Test sorting for the first two columns of the file to correctly
        # identify potential problems in sorting column 0.
        for column_i in [0, 1]:
            result = self.testIO_v2_multimeter.get_columns(sorting_column_indices=column_i)
            assert len(result) > 0
            assert all(np.diff(result[:, column_i]) >= 0)

        # Same procedure, supplying sorting with a list
        for column_i in [[0], [1]]:
            result = self.testIO_v2_multimeter.get_columns(sorting_column_indices=column_i)
            assert len(result) > 0
            assert all(np.diff(result[:, column_i[0]]) >= 0)

        # Same procedure, supplying sorting with a list of multiple columns of various priority
        # In this list, the last column has the highest priority.
        for column_i in [[0, 1], [1, 0], [1, 0, 2]]:
            result = self.testIO_v2_multimeter.get_columns(sorting_column_indices=column_i)
            assert len(result) > 0
            assert all(np.diff(result[:, column_i[-1]]) >= 0)

            # Additionally, create a checksum for the sorted and unsorted returns
            # to ensure all rows remain intact (same row sums)
            checksum = np.sort(np.sum(result, axis=1))
            result_unsorted = self.testIO_v2_multimeter.get_columns()
            checksum_unsorted = np.sort(np.sum(result_unsorted, axis=1))
            assert np.all(checksum == checksum_unsorted)

        # Same procedure, this time requesting only the highest priority column.
        for column_i in [[0, 1], [1, 0], [1, 0, 2]]:
            result = self.testIO_v2_multimeter.get_columns(column_indices=column_i[-1], sorting_column_indices=column_i)
            assert len(result) > 0
            assert all(np.diff(result[:, 0]) >= 0)

    def test_combined_selection_condition_sort(self):
        """
        get_columns() should select, sort, filter, or accept various forms of column/sort specification. This
        test will first test sorting and condition in isolation, then combine them in multiple combinations.
        """
        # Condition function used in tests
        odd_condition = lambda x: (int(x) % 2) == 1

        for cr in [self.testIO_v2_multimeter, self.testIO_v3_multimeter, self.testIO_v3_spikerecorder_precise]:
            # Sorting only: should sort by column values
            for sort_col_id in range(min(2, cr.data.shape[1])):
                sorted_data = cr.get_columns(sorting_column_indices=sort_col_id)
                # Make sure the sorted column is non-decreasing
                self.assertTrue(np.all(np.diff(sorted_data[:, sort_col_id]) >= 0))
                # Make sure the first column is correctly sorted according to sorting_column_indices
                np.testing.assert_array_equal(
                    sorted_data[:, 0],
                    cr.data[np.argsort(cr.data[:, sort_col_id]), 0])

            # Condition only: select rows with odd first column, if column exists
            cond_col_id = 0
            conditioned_data = cr.get_columns(condition=odd_condition, condition_column_index=cond_col_id)
            # Test all returned rows evaluate according the the odd condition
            for row in conditioned_data:
                self.assertTrue(odd_condition(row[cond_col_id]))
            # Test that a manual sorting yields the same array
            valid_indices = list(map(odd_condition, cr.data[:, cond_col_id]))
            np.testing.assert_array_equal(conditioned_data, cr.data[valid_indices ,:])

            # Define a few more complex parameters for the get_column function
            # to test for
            col_ids =[[0, 1] , [0, 1], [0, 1], [1, 0], [1, 0], [1, 0]]
            cond_col_ids = [0, 1, 1, 1, 1, 0]
            sort_col_ids = [0, [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]

            # Test each complex parameter set
            for col_ids, cond_col_id, sort_col_id in zip(col_ids, cond_col_ids, sort_col_ids):
                # Determine the column that is the highest priority for sorting
                if isinstance(sort_col_id, list):
                    sort_col_id_max_priority = sort_col_id[-1]
                else:
                    sort_col_id_max_priority = sort_col_id

                # Get the index of the condition column in the returned data
                cond_col_id_swapped = col_ids.index(cond_col_id)
                # Get the index of the sorting column in the returned data
                sort_col_id_max_priority_swapped = col_ids.index(sort_col_id_max_priority)

                sorted_conditioned_data = cr.get_columns(
                    column_indices=col_ids,
                    sorting_column_indices=sort_col_id,
                    condition=odd_condition, condition_column_index=cond_col_id)

                # Make sure that the sorted column is non-decreasing
                self.assertTrue(np.all(np.diff(sorted_conditioned_data[:, sort_col_id_max_priority_swapped]) >= 0))

                # Test all that all returned rows evaluate according to the odd condition
                for row in sorted_conditioned_data:
                    self.assertTrue(odd_condition(row[cond_col_id_swapped]))

                # Test that a manual sorting yields the same array
                # Note: In a scenario where sorting is done by one column only, the order of other columns may still be
                # arbitrary depending on the sorting algorithm. The current implementation of the ground truth sorting
                # below using the sorted function replicates the sorting done by the IO. However, changes to the IO may
                # break this test without violating the doc string. However, it will tell you that the IO chooses a
                # different sorting strategy.
                valid_indices = list(map(odd_condition, cr.data[:, cond_col_id]))
                np.testing.assert_array_equal(
                    sorted_conditioned_data,
                    np.array(sorted(cr.data[valid_indices , 0:2], key=lambda x : x[sort_col_id_max_priority]))[:, col_ids])

if __name__ == "__main__":
    unittest.main()
