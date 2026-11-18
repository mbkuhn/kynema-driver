# Module selection and execution

The driver coordinator is `driver::Simulation` in `src/Simulation.h`.
The shared module interface is `driver::Module` in `src/Module.h`. It does not
depend on MPI. Pass a communicator separately when registering a module;
`Simulation` keeps it and the module's timing data alongside the module.
Use `register_module()` to register a constructed module or construct a Tioga
adapter, and `register_module_no_tioga()` to construct a module without Tioga.
Initialization hooks use
`prepare_module_prolog()` and `prepare_module_epilog()`.
Every module implements `reset_states()`, called before advancing subsequent
nonlinear iterations. SGF and UGF currently keep their existing state handling
and implement this hook as a no-op.
Driver module adapters live in `src/modules/`. SGF owns its Tioga bridge in
`src/overset/SGFTiogaIface.H` and requires Tioga when built.
Include driver headers with their `src/` paths to distinguish the driver's
`Simulation.h` from the UGF library header of the same name.

Select modules in the `kynema_driver` YAML mapping. Existing SGF and UGF
configuration keys remain supported. FMB accepts either one input filename or
a list:

```yaml
kynema_driver:
  kynema_fmb_inp:
    - structure-a.yaml
    - structure-b.yaml
  fmb_procs: [1, 2]
  num_timesteps: 40
  nonlinear_iterations: 2
```

Input filenames are resolved relative to the working directory. SGF uses
`kynema_sgf_inp`; UGF uses `kynema_ugf_inp` and optionally `ugf_procs`. The
UGF list also accepts the existing `base_input_file`, `replace`, and `logfile`
mapping form. Module keys may be combined in one input file.

Run with MPI, optionally specifying the number of ranks for each module type:

```sh
mpiexec -n 4 kynema_driver --fmb 3 input.yaml
```

The options are `--sgf`, `--ugf`, and `--fmb`. Each selected type defaults
to all ranks. SGF uses ranks starting at zero; UGF and FMB each use a
contiguous range ending at the last rank. Different types may overlap.
Instances of the same type partition its range. If `ugf_procs` or `fmb_procs`
is supplied, its positive counts must sum to the rank count for that type;
otherwise ranks are divided as evenly as possible.

CMake options `KYNEMA_DRIVER_ENABLE_SGF` and `KYNEMA_DRIVER_ENABLE_UGF`
default to ON; `KYNEMA_DRIVER_ENABLE_FMB` defaults to OFF. Set an option to
OFF to build without that module's driver dependency, or enable FMB and
provide `KynemaFMB_DIR` to use it. Tioga and MPI remain driver dependencies.
A YAML file selecting a module disabled in the build is an error.

FMB's `GeneralStructureInterface` implements `driver::Module` directly;
the driver header `src/modules/KynemaFMB.h` is only a type alias, not a
timestepping wrapper. FMB always builds this interface, including for standalone
use, without a direct MPI dependency. A parallel NetCDF or HDF5 installation
may still bring in MPI; standalone FMB can use the serial versions. FMB installs
the shared interface header without a dependency on the driver library.
`KYNEMA_FMB_DRIVER_MODULE_DIR` only locates this header; it defaults to the
sibling `kynema-driver/src` directory and is not an integration toggle.
SGF and UGF still use adapters pending refactoring in their respective repositories.

Modules execute in the same timestep loop. Overset hooks are optional no-ops in
`Module`; only CFD modules opt in with `is_cfd()` and implement them.
CFD modules use `is_background_only()` to select background-only connectivity;
other CFD modules are generic. Kynema-SGF is a background-only CFD module whereas
Kynema-UGF is a generic CFD module because it can take on a near-body or background
role in an overset simulation. Non-CFD modules are neither category.
Overset activates only with at least two CFD modules or instances, counted
once per module communicator, regardless of rank count or non-CFD modules.
Connectivity and overset exchange skip non-CFD modules. SGF/UGF still run their
pre-advance preparation when overset is disabled.
FMB currently runs alongside the other modules but does not yet exchange forces
or displacements with them.

## Prescribed FMB loads

The driver-owned `Prescribed` module supplies six-component nodal forces
`[Fx, Fy, Fz, Mx, My, Mz]`. Select it with `prescribed_inp` (a filename or list)
alongside `kynema_fmb_inp`. Each prescribed YAML file names one FMB instance:

```yaml
target_fmb: 1
loads:
  - node: 0
    start_step: 22
    force: [0., 0., 50., 0., 0., 0.]
```

FMB instance IDs and start steps are one-based; node indices are zero-based.
Forces default to zero. Each entry replaces that node's force starting at the
specified step until the next entry for that node. Duplicate node/start-step
pairs are errors. Only one prescribed module may target each FMB instance.
It runs on its target's communicator, so `--fmb` and `fmb_procs` also determine
its ranks; there is no separate rank allocation or MPI transfer.
Use at least two nonlinear iterations: FMB predicts loads on the first
iteration and consumes the prescribed values on subsequent iterations.

FMB's `simulation.output_file` enables CSV output of every node's position and
quaternion, with the completed step and time, once per finalized timestep on
the instance's rank zero. The driver selects that rank and passes FMB a
plain output flag. Give separate output paths to separate FMB instances.

The `fmb-6dof-modular` regressions launch `kynema_driver` using these YAML inputs
and compare all output rows with `reference.csv` at absolute tolerance `1e-10`.
The saved reference was generated from the original standalone FMB sequence
(predict/advance, apply loads/reset/advance, output/finalize), for 40 steps of
0.125 seconds, applying 50 N vertically to the mass starting at step 22.
The Picard variant uses three nonlinear iterations and one extra correction.
