#ifndef DRIVER_PRESCRIBED_H
#define DRIVER_PRESCRIBED_H

#include "../Module.h"
#include "interfaces/generalized/coupling_data.hpp"
#include <yaml-cpp/yaml.h>

namespace driver {

class Prescribed : public Module
{
public:
    Prescribed(
        const YAML::Node& input, const std::string& target_name,
        double dt, size_t num_nodes);
    bool is_fixed_timestep_size() override { return true; }
    int time_index() override { return m_step; }
    std::string identifier() override { return m_name; }
    nlohmann::json outputs() const override { return m_outputs; }

protected:
    void init_prolog(bool) override {}
    void init_epilog() override {}
    void prepare_module_prolog() override {}
    void prepare_module_epilog() override {}
    double get_time() override { return m_time; }
    double get_timestep_size() override { return m_dt; }
    void set_timestep_size(double dt) override;
    void set_inputs(size_t) override {}
    void predict_inputs(size_t) override {}
    void reset_states() override {}
    void advance_states(size_t) override {}
    void calculate_outputs(size_t) override;
    void additional_picard_iterations(int) override {}
    void finalize_time_step() override;
    void dump_simulation_time() override {}

private:
    struct Load
    {
        int start_step;
        size_t node;
        std::array<double, 6> force;
    };

    std::string m_name;
    double m_dt;
    double m_time{0.};
    int m_step{0};
    std::vector<Load> m_loads;
    kynema_fmb::interfaces::StructuralInputs m_outputs;
};

} // namespace driver
#endif
