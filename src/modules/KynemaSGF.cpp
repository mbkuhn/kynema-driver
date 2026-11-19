#include "KynemaSGF.h"
#include "src/incflo.H"
#include "src/CFDSim.H"
#include "src/core/SimTime.H"
#include "src/utilities/console_io.H"
#include "AMReX.H"
#include "AMReX_ParmParse.H"

#include "tioga.h"

namespace driver {

// Move process-wide AMReX setup and teardown behind an SGF-owned runtime API.
void KynemaSGFInterface::initialize(
    MPI_Comm comm, const std::string& inpfile, std::ofstream& out)
{
    int argc = 2;
    char** argv = new char*[argc];

    const char* exename = "kynema_sgf";
    argv[0] = const_cast<char*>(exename);
    argv[1] = const_cast<char*>(inpfile.c_str());

    amrex::Initialize(
        argc, argv, true, comm,
        []() {
            amrex::ParmParse pp("amrex");
            // Set the defaults so that we throw an exception instead of
            // attempting to generate backtrace files. However, if the user has
            // explicitly set these options in their input files respect those
            // settings.
            if (!pp.contains("throw_exception")) pp.add("throw_exception", 1);
            if (!pp.contains("signal_handling")) pp.add("signal_handling", 0);
        },
        out, out);

    kynema_sgf::io::print_banner(comm, amrex::OutStream());

    delete[] argv;
}

void KynemaSGFInterface::finalize() { amrex::Finalize(); }

KynemaSGFInterface::KynemaSGFInterface(
    const std::vector<std::string>& cell_vars,
    const std::vector<std::string>& node_vars,
    TIOGA::tioga& tg)
    : m_incflo()
    , m_tgiface(m_incflo.sim(), tg)
    , m_cell_vars(cell_vars)
    , m_node_vars(node_vars)
{}

KynemaSGFInterface::~KynemaSGFInterface() = default;

// Expose mesh/module initialization and prepare_for_time_integration through an
// SGF-owned preparation API. Map mesh and exchange readiness in driver traits
// so the solver never needs the driver's PreparationContext.
PreparationStage KynemaSGFInterface::prepare(const PreparationContext& context)
{
    if (context.stage == PreparationStage::pending) {
        if (context.use_overset) m_incflo.sim().activate_overset();
        m_incflo.init_mesh();
        m_incflo.init_kynema_sgf_modules();
        return PreparationStage::mesh_ready;
    }
    if (context.stage == PreparationStage::mesh_ready && context.meshes_ready)
        return PreparationStage::exchange_ready;
    if (context.stage == PreparationStage::exchange_ready &&
        context.solution_exchanged) {
        m_incflo.prepare_for_time_integration();
        return PreparationStage::complete;
    }
    return context.stage;
}

// Expose this first-iteration timestep setup as SGF's native set_inputs().
void KynemaSGFInterface::set_inputs(std::size_t inonlin)
{
    if (inonlin < 1) {
        m_incflo.sim().time().new_timestep();
        m_incflo.regrid_and_update();
        m_incflo.compute_dt();
    }
}

// Expose the two pre-advance phases separately: stage 2 must still run after
// connectivity, even when Tioga is inactive.
void KynemaSGFInterface::predict_inputs(std::size_t inonlin)
{
    if (inonlin < 1) {
        m_incflo.pre_advance_stage1();
        m_stage2_pending = true;
    }
}

void KynemaSGFInterface::prepare_solution_exchange()
{
    if (m_stage2_pending) {
        m_incflo.pre_advance_stage2();
        m_stage2_pending = false;
    }
}

// Expose these accessors on the SGF interface instead of reaching into SimTime.
double KynemaSGFInterface::get_time() { return m_incflo.time().new_time(); }

double KynemaSGFInterface::get_timestep_size()
{
    return m_incflo.time().delta_t();
}

void KynemaSGFInterface::set_timestep_size(const double dt)
{
    m_incflo.sim().time().delta_t() = dt;
}

bool KynemaSGFInterface::is_fixed_timestep_size()
{
    return (!m_incflo.sim().time().adaptive_timestep());
}

// Expose do_advance and post_advance_work as advance_states and
// finalize_time_step on the native interface.
void KynemaSGFInterface::advance_states(std::size_t inonlin)
{
    m_incflo.do_advance(inonlin);
}

void KynemaSGFInterface::finalize_time_step() { m_incflo.post_advance_work(); }

// Keep these forwarding hooks and registered field lists with SGF's
// optional Tioga bridge when the native interface moves into SGF.
void KynemaSGFInterface::pre_overset_conn_work()
{
    m_tgiface.pre_overset_conn_work();
}

void KynemaSGFInterface::post_overset_conn_work()
{
    m_tgiface.post_overset_conn_work();
}

void KynemaSGFInterface::register_solution()
{
    m_tgiface.register_solution(m_cell_vars, m_node_vars);
}

void KynemaSGFInterface::update_solution() { m_tgiface.update_solution(); }

int KynemaSGFInterface::overset_update_interval()
{
    const int regrid_int = m_incflo.sim().time().regrid_interval();
    return regrid_int > 0 ? regrid_int : 100000000;
}

int KynemaSGFInterface::time_index()
{
    return m_incflo.sim().time().time_index();
}

} // namespace driver
