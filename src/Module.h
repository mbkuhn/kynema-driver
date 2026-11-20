#ifndef DRIVER_MODULE_H
#define DRIVER_MODULE_H

#include <cstddef>
#include <string>
#include <nlohmann/json.hpp>

namespace driver {

enum class PreparationStage { pending, mesh_ready, exchange_ready, complete };

struct PreparationContext
{
    PreparationStage stage;
    bool use_overset;
    bool meshes_ready;
    bool solution_exchanged;
};

class Module
{
public:
    virtual ~Module() = default;

    virtual bool is_cfd() { return false; }
    virtual bool is_background_only() { return false; }
    virtual bool is_fixed_timestep_size() = 0;
    virtual int overset_update_interval() { return 100000000; }
    virtual int time_index() = 0;
    virtual std::string identifier() { return "Module"; }
    virtual int get_ncomps() { return 0; }

    // Return the current stage unchanged while waiting for a dependency.
    virtual PreparationStage prepare(const PreparationContext&) = 0;

    // Null means no payload; the first non-null initialization payload is sent
    // once.
    virtual nlohmann::json initialization_data() const { return nullptr; }
    virtual void receive_initialization_data(const nlohmann::json&) {}

    virtual double get_time() = 0;
    virtual double get_timestep_size() = 0;
    virtual void set_timestep_size(double) = 0;
    virtual void additional_picard_iterations(int) = 0;
    virtual void finalize_time_step() = 0;
    virtual void pre_overset_conn_work() {}
    virtual void post_overset_conn_work() {}
    virtual void register_solution() {}
    virtual void prepare_solution_exchange() {}
    virtual void update_solution() {}
    virtual void dump_simulation_time() = 0;

    virtual void set_inputs(std::size_t inonlin) = 0;
    virtual void predict_inputs(std::size_t inonlin) = 0;
    virtual void reset_states() = 0;
    virtual void advance_states(std::size_t inonlin) = 0;
    virtual void calculate_outputs(std::size_t inonlin) = 0;

    virtual void receive_inputs(const nlohmann::json&) {}
    virtual nlohmann::json outputs() const { return nullptr; }
};

} // namespace driver
#endif /* DRIVER_MODULE_H */
