#ifndef KYNEMAUGF_H
#define KYNEMAUGF_H

#include <vector>
#include <string>

#include "Simulation.h"
#include "stk_util/parallel/Parallel.hpp"
#include "yaml-cpp/yaml.h"
#include "CfdModuleAdapter.h"

namespace TIOGA {
class tioga;
}

namespace driver {

// Temporary shim until UGF exports a native lifecycle interface like FMB.
class KynemaUGFInterface
{
private:
    YAML::Node m_doc;
    sierra::kynema_ugf::Simulation m_sim;
    std::vector<std::string> m_fnames;
    int m_ncomps;
    int m_id;
    bool m_stage2_pending{false};
    size_t m_stage2_iteration{0};
    void load_mesh(bool use_overset);

public:
    static void initialize();
    static void finalize();
    static std::string change_file_name_suffix(
        std::string inpfile, std::string suffix, int index = -1)
    {
        int extloc = inpfile.rfind(".");
        std::string logfile = inpfile;
        if (index >= 0) {
            suffix = "_" + std::to_string(index) + suffix;
        }
        if (extloc != static_cast<int>(std::string::npos)) {
            logfile = inpfile.substr(0, extloc) + suffix;
        }
        return logfile;
    }
    explicit KynemaUGFInterface(
        int id,
        stk::ParallelMachine comm,
        const YAML::Node& inp_yaml,
        const std::string& logfile,
        const std::vector<std::string>& fnames,
        TIOGA::tioga& tg);
    ~KynemaUGFInterface();
    bool is_fixed_timestep_size();
    int overset_update_interval();
    int time_index();
    std::string identifier() { return ("Kynema-UGF-" + std::to_string(m_id)); }
    int get_ncomps() { return m_ncomps; }

    // UGF should publish or consume JSON here when it has coupling data.
    nlohmann::json initialization_data() const { return nullptr; }
    void receive_initialization_data(const nlohmann::json&) {}
    void receive_inputs(const nlohmann::json&) {}
    nlohmann::json outputs() const { return nullptr; }

    PreparationStage prepare(const PreparationContext&);
    void set_inputs(std::size_t inonlin);
    void predict_inputs(std::size_t inonlin);
    void reset_states() {} // UGF must expose state restore for coupled retries.
    void advance_states(std::size_t inonlin);
    void calculate_outputs(std::size_t) {}
    double get_time();
    double get_timestep_size();
    void set_timestep_size(double);
    void additional_picard_iterations(int);
    void finalize_time_step();
    void pre_overset_conn_work();
    void post_overset_conn_work();
    void prepare_solution_exchange();
    void register_solution();
    void update_solution();
    void dump_simulation_time();
};

template <>
struct ModuleAdapterTraits<KynemaUGFInterface>
    : CfdModuleAdapterTraits<KynemaUGFInterface>
{};

using KynemaUGF = CfdModuleAdapter<KynemaUGFInterface, false>;

} // namespace driver

#endif /* KYNEMAUGF_H */
