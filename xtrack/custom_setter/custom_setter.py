import numpy as np

import xobjects as xo

source = """

void get_values_at_offsets(
    CustomSetterData data,
    int8_t* buffer,
    double* out){

    int64_t num_offsets = CustomSetterData_len_offsets(data);

    int64_t iout = 0;
    for (int64_t ii = 0; ii < num_offsets; ii++) { //vectorize_over ii num_offsets
        int64_t offs = CustomSetterData_get_offsets(data, ii);

        double val = *((double*)(buffer + offs));
        out[iout] = val;
        iout++;
    } //end_vectorize
}

void set_values_at_offsets(
    CustomSetterData data,
    int8_t* buffer,
    double* input){

    int64_t num_offsets = CustomSetterData_len_offsets(data);

    int64_t iin = 0;
    for (int64_t ii = 0; ii < num_offsets; ii++) {  //vectorize_over ii num_offsets
        int64_t offs = CustomSetterData_get_offsets(data, ii);

        double val = input[iin];
        *((double*)(buffer + offs)) = val;
        iin++;
    } //end_vectorize
}

"""

class CustomSetter(xo.HybridClass):
    _xofields = {
        'offsets': xo.Int64[:],
    }

    _extra_c_sources = [
        source,
    ]

    _kernels = {
        'get_values_at_offsets': xo.Kernel(
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Float64, pointer=True, name='out'),
            ],
        ),
        'set_values_at_offsets': xo.Kernel(
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Float64, pointer=True, name='input'),
            ],
        ),
    }

    def __init__(self, tracker, elements, field, index=None):

        context = tracker._context

        tracker_buffer = tracker._buffer
        line = tracker.line
        assert np.all([line[nn]._buffer is tracker_buffer for nn in elements])
        offsets = [_extract_offset(line[nn], field, index) for nn in elements]

        self.xoinitialize(_context=context, offsets=offsets)
        self.compile_kernels(only_if_needed=True)

        self.offsets = offsets
        self.tracker = tracker
        self._tracker_buffer = tracker_buffer

    def get_values(self):
        out = np.zeros(len(self.offsets), dtype=np.float64)
        kernel = self._context.kernels.get_values_at_offsets
        kernel.description.n_threads = len(self.offsets)
        kernel(data=self, buffer=self._tracker_buffer.buffer, out=out)
        return out

    def set_values(self, values):
        kernel = self._context.kernels.set_values_at_offsets
        kernel.description.n_threads = len(self.offsets)
        kernel(data=self, buffer=self._tracker_buffer.buffer, input=values)


def _extract_offset(obj, field_name, index):
    if index is None:
        assert isinstance(getattr(obj, field_name), np.float64), (
            'Only float64 fields are supported for now')
        return obj._xobject._get_offset(field_name)
    else:
        assert getattr(obj._xobject, field_name)._itemtype is xo.Float64, (
            'Only Float64 arrays are supported for now')
        return getattr(obj._xobject, field_name)._get_offset(index)