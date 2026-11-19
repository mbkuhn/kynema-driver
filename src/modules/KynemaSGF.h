#ifndef KYNEMASGF_H
#define KYNEMASGF_H

#include "CfdModuleAdapter.h"
#include "src/incflo.H"
#include "src/overset/SGFTiogaIface.H"

namespace TIOGA {
class tioga;
}

namespace driver {

// Temporary shim until SGF exports a native lifecycle interface like FMB.
class KynemaSGFInterface
{
private:
    // Keep this declaration order if moved into SGF: the Tioga bridge borrows
    // m_incflo.sim() and must be destroyed before m_incflo.
    incflo m_incflo;
    kynema_sgf::SGFTiogaIface m_tgiface;
    std::vector<std::string> m_cell_vars;
    std::vector<std::string> m_node_vars;
    bool m_stage2_pending{false};

public:
    static void
    initialize(MPI_Comm comm, const std::string& inpfile, std::ofstream& out);
    static void finalize();
    explicit KynemaSGFInterface(
        const std::vector<std::string>&,
        const std::vector<std::string>&,
        TIOGA::tioga&);
    ~KynemaSGFInterface();
    bool is_fixed_timestep_size();
    int overset_update_interval();
    int time_index();
    std::string identifier() { return "Kynema-SGF"; }

    // SGF should publish or consume JSON here when it has coupling data.
    nlohmann::json initialization_data() const { return nullptr; }
    void receive_initialization_data(const nlohmann::json&) {}
    void receive_inputs(const nlohmann::json&) {}
    nlohmann::json outputs() const { return nullptr; }

    PreparationStage prepare(const PreparationContext&);
    void set_inputs(std::size_t inonlin);
    void predict_inputs(std::size_t inonlin);
    void reset_states() {} // SGF must expose state restore for coupled retries.
    void advance_states(std::size_t inonlin);
    void calculate_outputs(std::size_t) {}
    double get_time();
    double get_timestep_size();
    void set_timestep_size(double);
    void additional_picard_iterations(int) {}
    void finalize_time_step();
    void pre_overset_conn_work();
    void post_overset_conn_work();
    void prepare_solution_exchange();
    void register_solution();
    void update_solution();
    void dump_simulation_time() {}
};

template <>
struct ModuleAdapterTraits<KynemaSGFInterface>
    : CfdModuleAdapterTraits<KynemaSGFInterface>
{};

using KynemaSGF = CfdModuleAdapter<KynemaSGFInterface, true>;

} // namespace driver

#endif
