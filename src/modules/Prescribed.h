#ifndef DRIVER_PRESCRIBED_H
#define DRIVER_PRESCRIBED_H

#include "KynemaFMB.h"
#include <yaml-cpp/yaml.h>

namespace driver {

class Prescribed : public Module
{
public:
    Prescribed(const YAML::Node& input, KynemaFMB& target);
    bool is_fixed_timestep_size() override { return true; }
    int time_index() override { return m_step; }
    std::string identifier() override { return "Prescribed-" + m_target.identifier(); }

protected:
    void init_prolog(bool) override {}
    void init_epilog() override {}
    void prepare_module_prolog() override {}
    void prepare_module_epilog() override {}
    double get_time() override { return m_time; }
    double get_timestep_size() override { return m_dt; }
    void set_timestep_size(double dt) override;
    void set_inputs(size_t) override;
    void predict_inputs(size_t) override {}
    void reset_states() override {}
    void advance_states(size_t) override {}
    void calculate_outputs(size_t) override {}
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

    KynemaFMB& m_target;
    double m_dt;
    double m_time{0.};
    int m_step{0};
    std::vector<Load> m_loads;
    std::vector<std::array<double, 6>> m_forces;
};

} // namespace driver
#endif
