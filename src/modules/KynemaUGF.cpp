#include "KynemaUGF.h"
#include "KynemaUGFEnv.h"
#include "Realm.h"
#include "TimeIntegrator.h"
#include "overset/ExtOverset.h"
#include "overset/TiogaRef.h"

#include "Kokkos_Core.hpp"
#include "tioga.h"
#include "HypreNGP.h"

namespace driver {

// Move process-wide Kokkos/Hypre setup and teardown behind a UGF-owned runtime
// API.
void KynemaUGFInterface::initialize()
{
    Kokkos::initialize();
    // Hypre initialization
    kynema_ugf_hypre::hypre_initialize();
    kynema_ugf_hypre::hypre_set_params();
}

void KynemaUGFInterface::finalize()
{
    // Hypre cleanup
    kynema_ugf_hypre::hypre_finalize();

    if (Kokkos::is_initialized()) {
        Kokkos::finalize();
    }
}

// Let UGF own communicator, log, and Tioga setup rather than mutating its
// environment singleton and TiogaRef from the driver.
KynemaUGFInterface::KynemaUGFInterface(
    int id,
    stk::ParallelMachine comm,
    const YAML::Node& inp_yaml,
    const std::string& logfile,
    const std::vector<std::string>& fnames,
    TIOGA::tioga& tg)
    : m_doc(inp_yaml), m_sim(m_doc), m_fnames(fnames), m_id(id)
{
    auto& env = sierra::kynema_ugf::KynemaUGFEnv::self();
    env.parallelCommunicator_ = comm;
    MPI_Comm_size(comm, &env.pSize_);
    MPI_Comm_rank(comm, &env.pRank_);

    ::tioga_kynema_ugf::TiogaRef::self(&tg);

    env.set_log_file_stream(logfile);
}

KynemaUGFInterface::~KynemaUGFInterface() = default;

void KynemaUGFInterface::load_mesh(bool use_overset)
{
    // Dump the input yaml to the start of the logfile
    // before the kynema_ugf banner
    auto& env = sierra::kynema_ugf::KynemaUGFEnv::self();
    env.kynema_ugfOutputP0() << std::string(20, '#') << " INPUT FILE START "
                             << std::string(20, '#') << std::endl;
    sierra::kynema_ugf::KynemaUGFParsingHelper::emit(
        *env.kynema_ugfLogStream_, m_doc);
    env.kynema_ugfOutputP0() << std::string(20, '#') << " INPUT FILE END   "
                             << std::string(20, '#') << std::endl;

    m_sim.load(m_doc);
    if (m_sim.timeIntegrator_->overset_ != nullptr)
        m_sim.timeIntegrator_->overset_->set_multi_solver_mode(use_overset);
    m_sim.breadboard();
    m_sim.init_prolog();
}

// Wrap load/breadboard and init_prolog/init_epilog in UGF-owned preparation
// methods. Driver traits should map mesh and exchange readiness to those calls
// without exposing PreparationContext or walking UGF realms from the driver.
PreparationStage KynemaUGFInterface::prepare(const PreparationContext& context)
{
    if (context.stage == PreparationStage::pending) {
        load_mesh(context.use_overset);
        return PreparationStage::mesh_ready;
    }
    if (context.stage == PreparationStage::mesh_ready && context.meshes_ready) {
        m_sim.init_epilog();
        m_sim.timeIntegrator_->prepare_for_time_integration();
        return PreparationStage::exchange_ready;
    }
    if (context.stage == PreparationStage::exchange_ready &&
        context.solution_exchanged) {
        for (auto* realm : m_sim.timeIntegrator_->realmVec_)
            realm->output_converged_results();
        return PreparationStage::complete;
    }
    return context.stage;
}

// Expose prepare_time_step as set_inputs and the first pre-realm advance phase
// as predict_inputs. Keep the second phase after connectivity.
void KynemaUGFInterface::set_inputs(std::size_t inonlin)
{
    m_sim.timeIntegrator_->prepare_time_step(inonlin);
}

void KynemaUGFInterface::predict_inputs(std::size_t inonlin)
{
    m_sim.timeIntegrator_->pre_realm_advance_stage1(inonlin);
    m_stage2_iteration = inonlin;
    m_stage2_pending = true;
}

void KynemaUGFInterface::prepare_solution_exchange()
{
    if (m_stage2_pending) {
        m_sim.timeIntegrator_->pre_realm_advance_stage2(m_stage2_iteration);
        m_stage2_pending = false;
    }
}

// Expose time and timestep accessors on the native interface rather than
// dereferencing timeIntegrator_ or reading timeStepCount_ from the driver.
double KynemaUGFInterface::get_time()
{
    return m_sim.timeIntegrator_->get_current_time();
}

double KynemaUGFInterface::get_timestep_size()
{
    return m_sim.timeIntegrator_->get_time_step();
}

void KynemaUGFInterface::set_timestep_size(const double dt)
{
    m_sim.timeIntegrator_->set_time_step(dt);
}

bool KynemaUGFInterface::is_fixed_timestep_size()
{
    return m_sim.timeIntegrator_->get_is_fixed_time_step();
}

// Move the realm loops behind native advance_states and Picard hooks. Expose
// post_realm_advance as finalize_time_step on that same interface.
void KynemaUGFInterface::advance_states(std::size_t /*inonlin*/)
{
    for (auto* realm : m_sim.timeIntegrator_->realmVec_) {
        realm->advance_time_step();
        realm->process_multi_physics_transfer();
    }
}

void KynemaUGFInterface::additional_picard_iterations(const int n)
{
    for (auto* realm : m_sim.timeIntegrator_->realmVec_)
        realm->nonlinear_iterations(n);
}

void KynemaUGFInterface::finalize_time_step()
{
    m_sim.timeIntegrator_->post_realm_advance();
}

// Publish the overset hooks and registered field component count from UGF's
// bridge instead of accessing timeIntegrator_->overset_ from the driver.
void KynemaUGFInterface::pre_overset_conn_work()
{
    m_sim.timeIntegrator_->overset_->pre_overset_conn_work();
}

void KynemaUGFInterface::post_overset_conn_work()
{
    m_sim.timeIntegrator_->overset_->post_overset_conn_work();
}

void KynemaUGFInterface::register_solution()
{
    m_ncomps = m_sim.timeIntegrator_->overset_->register_solution(m_fnames);
}

void KynemaUGFInterface::update_solution()
{
    m_sim.timeIntegrator_->overset_->update_solution();
}

int KynemaUGFInterface::overset_update_interval()
{
    for (auto& realm : m_sim.timeIntegrator_->realmVec_) {
        if (realm->does_mesh_move()) {
            return 1;
        }
    }
    return 100000000;
}

int KynemaUGFInterface::time_index()
{
    return m_sim.timeIntegrator_->timeStepCount_;
}

void KynemaUGFInterface::dump_simulation_time()
{
    for (auto& realm : m_sim.timeIntegrator_->realmVec_) {
        realm->dump_simulation_time();
    }
}

} // namespace driver
