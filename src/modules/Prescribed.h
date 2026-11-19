#ifndef DRIVER_PRESCRIBED_H
#define DRIVER_PRESCRIBED_H

#include "../Module.h"
#include <vector>
#include <yaml-cpp/yaml.h>

namespace driver {

class Prescribed : public Module
{
public:
    explicit Prescribed(const YAML::Node& input) : m_input(input) {}
    bool is_fixed_timestep_size() override { return true; }
    int time_index() override { return m_step; }
    std::string identifier() override { return m_name; }
    nlohmann::json outputs() const override;

protected:
    PreparationStage prepare(const PreparationContext&) override;
    void receive_initialization_data(const nlohmann::json& data) override
    {
        m_initialization = data;
    }
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
        std::vector<double> force;
    };

    YAML::Node m_input;
    nlohmann::json m_initialization;
    std::string m_name{"Prescribed"};
    std::string m_field;
    int m_version{0};
    double m_dt{0.};
    double m_time{0.};
    int m_step{0};
    std::vector<Load> m_loads;
    std::vector<std::vector<double>> m_values;
};

} // namespace driver
#endif
