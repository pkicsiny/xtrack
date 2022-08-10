# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from pathlib import Path
import numpy as np

import xobjects as xo
import xpart as xp

from xobjects.dressed_struct import _build_xofields_dict

from .general import _pkg_root
from .interal_record import RecordIdentifier, RecordIndex, generate_get_record

start_per_part_block = """
   int64_t const n_part = LocalParticle_get__num_active_particles(part0); //only_for_context cpu_serial cpu_openmp
   #pragma omp parallel for                                       //only_for_context cpu_openmp
   for (int jj=0; jj<n_part; jj+=!!CHUNK_SIZE!!){                 //only_for_context cpu_serial cpu_openmp
    //#pragma omp simd
    for (int iii=0; iii<!!CHUNK_SIZE!!; iii++){                   //only_for_context cpu_serial cpu_openmp
      int const ii = iii+jj;                                      //only_for_context cpu_serial cpu_openmp
      if (ii<n_part){                                             //only_for_context cpu_serial cpu_openmp

        LocalParticle lpart = *part0;//only_for_context cpu_serial cpu_openmp
        LocalParticle* part = &lpart;//only_for_context cpu_serial cpu_openmp
        part->ipart = ii;            //only_for_context cpu_serial cpu_openmp

        LocalParticle* part = part0;//only_for_context opencl cuda
""".replace("!!CHUNK_SIZE!!", "128")

end_part_part_block = """
     } //only_for_context cpu_serial cpu_openmp
    }  //only_for_context cpu_serial cpu_openmp
   }   //only_for_context cpu_serial cpu_openmp
"""

def _handle_per_particle_blocks(sources):

    out = []
    for ii, ss in enumerate(sources):
        if isinstance(ss, Path):
            with open(ss, 'r') as fid:
                strss = fid.read()
        else:
            strss = ss

        if '//start_per_particle_block' in strss:

            lines = strss.splitlines()
            for ill, ll in enumerate(lines):
                if '//start_per_particle_block' in ll:
                    lines[ill] = start_per_part_block
                if '//end_per_particle_block' in ll:
                    lines[ill] = end_part_part_block

            # TODO: this is very dirty, just for check!!!!!
            out.append('\n'.join(lines))
        else:
            out.append(ss)

    return out

def _generate_per_particle_kernel_from_local_particle_function(
                                                element_name, kernel_name,
                                                local_particle_function_name,
                                                additional_args=[]):

    if len(additional_args) > 0:
        add_to_signature = ", ".join([
            f"{' /*gpuglmem*/ ' if arg.pointer else ''} {arg.get_c_type()} {arg.name}"
                for arg in additional_args]) + ", "
        add_to_call = ", " + ", ".join(f"{arg.name}" for arg in additional_args)

    source = ('''
            /*gpukern*/
            '''
            f'void {kernel_name}(\n'
            f'               {element_name}Data el,\n'
'''
                             ParticlesData particles,
'''
            f'{(add_to_signature if len(additional_args) > 0 else "")}'
'''
                             int64_t flag_increment_at_element,
                /*gpuglmem*/ int8_t* io_buffer){
            LocalParticle lpart;
            lpart.io_buffer = io_buffer;

            int64_t part_id = 0;                    //only_for_context cpu_serial cpu_openmp
            int64_t part_id = blockDim.x * blockIdx.x + threadIdx.x; //only_for_context cuda
            int64_t part_id = get_global_id(0);                    //only_for_context opencl

            int64_t part_capacity = ParticlesData_get__capacity(particles);
            if (part_id<part_capacity){
                Particles_to_LocalParticle(particles, &lpart, part_id);
                if (check_is_active(&lpart)>0){
'''
            f'      {local_particle_function_name}(el, &lpart{(add_to_call if len(additional_args) > 0 else "")});\n'
'''
                }
                if (check_is_active(&lpart)>0 && flag_increment_at_element){
                        increment_at_element(&lpart);
                }
            }
        }
''')
    return source

class MetaBeamElement(xo.MetaDressedStruct):

    def __new__(cls, name, bases, data):
        XoStruct_name = name+'Data'

        xofields = _build_xofields_dict(bases, data)
        data = data.copy()
        data['_xofields'] = xofields

        if '_internal_record_class' in data.keys():
            data['_xofields']['_internal_record_id'] = RecordIdentifier
            if '_skip_in_to_dict' not in data.keys():
                data['_skip_in_to_dict'] = []
            data['_skip_in_to_dict'].append('_internal_record_id')

            sources_for_internal_record = []
            sources_for_internal_record.extend(
                xo.context.sources_from_classes(xo.context.sort_classes(
                                            [data['_internal_record_class'].XoStruct])))
            sources_for_internal_record.append(
                generate_get_record(ele_classname=XoStruct_name,
                    record_classname=data['_internal_record_class'].XoStruct.__name__))

            data['extra_sources'] = sources_for_internal_record + data['extra_sources']

        new_class = xo.MetaDressedStruct.__new__(cls, name, bases, data)
        XoStruct = new_class.XoStruct

        new_class.per_particle_kernels_source = _generate_per_particle_kernel_from_local_particle_function(
            element_name=name, kernel_name=name+'_track_particles',
            local_particle_function_name=name+'_track_local_particle')

        new_class._track_kernel_name = f'{name}_track_particles'
        new_class.per_particle_kernels_description = {new_class._track_kernel_name:
            xo.Kernel(args=[xo.Arg(XoStruct, name='el'),
                        xo.Arg(xp.Particles.XoStruct, name='particles'),
                        xo.Arg(xo.Int64, name='flag_increment_at_element'),
                        xo.Arg(xo.Int8, pointer=True, name="io_buffer")])}

        XoStruct._DressingClass = new_class

        if '_internal_record_class' in data.keys():
            new_class.XoStruct._internal_record_class = data['_internal_record_class']
            new_class._internal_record_class = data['_internal_record_class']


        if 'per_particle_kernels' in data.keys():
            for nn, kk in data['per_particle_kernels'].items():
                new_class.per_particle_kernels_source += ('\n' +
                    _generate_per_particle_kernel_from_local_particle_function(
                        element_name=name, kernel_name=nn,
                        local_particle_function_name=kk.c_name,
                        additional_args=kk.args))
                setattr(new_class, nn, PerParticleMethodDescriptor(kernel_name=nn))

                new_class.per_particle_kernels_description.update(
                    {nn:
                        xo.Kernel(args=[xo.Arg(new_class.XoStruct, name='el'),
                        xo.Arg(xp.Particles.XoStruct, name='particles')]
                        + kk.args + [
                        xo.Arg(xo.Int64, name='flag_increment_at_element'),
                        xo.Arg(xo.Int8, pointer=True, name="io_buffer")])}
                )

        return new_class

class BeamElement(metaclass=MetaBeamElement):

    iscollective = None

    def init_pipeline(self,pipeline_manager,name,partners_names=[]):
        self._pipeline_manager = pipeline_manager
        self.name = name
        self.partners_names = partners_names

    def compile_per_particle_kernels(self, save_source_as=None):
        context = self._buffer.context

        sources = []

        # Local particles
        sources.append(xp.gen_local_particle_api())

        # Tracker auxiliary functions
        sources.append(_pkg_root.joinpath("tracker_src/tracker.h"))

        # Internal recording
        sources.append(RecordIdentifier._gen_c_api())
        sources += RecordIdentifier.extra_sources
        sources.append(RecordIndex._gen_c_api())
        sources += RecordIndex.extra_sources

        sources += self.XoStruct.extra_sources
        sources.append(self.per_particle_kernels_source)

        sources = _handle_per_particle_blocks(sources)

        context.add_kernels(sources=sources,
                kernels=self.per_particle_kernels_description,
                save_source_as=save_source_as)

    def track(self, particles, increment_at_element=False):

        context = self._buffer.context
        if not hasattr(self, '_track_kernel'):
            if self._track_kernel_name not in context.kernels.keys():
                self.compile_per_particle_kernels()
            self._track_kernel = context.kernels[self._track_kernel_name]

        if hasattr(self, 'io_buffer') and self.io_buffer is not None:
            io_buffer_arr = self.io_buffer.buffer
        else:
            io_buffer_arr=context.zeros(1, dtype=np.int8) # dummy

        self._track_kernel.description.n_threads = particles._capacity
        self._track_kernel(el=self._xobject, particles=particles,
                           flag_increment_at_element=increment_at_element,
                           io_buffer=io_buffer_arr)

    def _arr2ctx(self, arr):
        ctx = self._buffer.context

        if isinstance(arr, list):
            arr = np.array(arr)

        if np.isscalar(arr):
            if hasattr(arr, 'item'):
                return arr.item()
            else:
                return arr
        elif isinstance(arr, ctx.nplike_array_type):
            return arr
        elif isinstance(arr, np.ndarray):
            return ctx.nparray_to_context_array(arr)
        else:
            raise ValueError("Invalid array type")


class PerParticleMethod:

    def __init__(self, kernel, element):
        self.kernel = kernel
        self.element = element

    def __call__(self, particles, increment_at_element=False, **kwargs):

        if hasattr(self, 'io_buffer') and self.io_buffer is not None:
            io_buffer_arr = self.io_buffer.buffer
        else:
            context = self.kernel.context
            io_buffer_arr=context.zeros(1, dtype=np.int8) # dummy

        self.kernel.description.n_threads = particles._capacity
        self.kernel(el=self.element._xobject, particles=particles,
                           flag_increment_at_element=increment_at_element,
                           io_buffer=io_buffer_arr,
                           **kwargs)

class PerParticleMethodDescriptor:

    def __init__(self, kernel_name):
        self.kernel_name = kernel_name

    def __get__(self, instance, owner):
        context = instance._buffer.context
        if not hasattr(instance, '_track_kernel'):
            if instance._track_kernel_name not in context.kernels.keys():
                instance.compile_per_particle_kernels()
            instance._track_kernel = context.kernels[instance._track_kernel_name]

        return PerParticleMethod(kernel=context.kernels[self.kernel_name],
                                 element=instance)
